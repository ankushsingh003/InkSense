"""InkSense model: 3D-CNN encoder + Transformer context + 2D decoder.

Input : (B, 1, Z, H, W)  - a stack of Z surface slices of a CT volume
Output: (B, 1, H, W)     - ink logits per pixel

Design notes
------------
* The 3D encoder downsamples Z *and* H/W by 8 in total, so global self-attention runs on
  (H/8)*(W/8) tokens (1,024 tokens for a 256x256 tile) instead of H*W (65,536), which keeps
  attention memory small enough for a consumer GPU.
* A 2D sin-cos positional encoding gives the Transformer spatial awareness.
* The decoder upsamples back to full resolution and fuses a high-resolution skip connection
  (mean over Z of the stem features) so thin strokes are not lost by the downsampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def sincos_2d(h, w, dim, device, dtype):
    """Fixed 2D sin-cos positional encoding, shape (h*w, dim). `dim` must be divisible by 4."""
    assert dim % 4 == 0, "model_dim must be divisible by 4"
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter))
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * omega[None]  # (h, q)
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * omega[None]  # (w, q)
    ys = torch.cat([ys.sin(), ys.cos()], dim=1)  # (h, dim/2)
    xs = torch.cat([xs.sin(), xs.cos()], dim=1)  # (w, dim/2)
    pos = torch.cat([ys[:, None, :].expand(h, w, -1), xs[None, :, :].expand(h, w, -1)], dim=2)
    return pos.reshape(h * w, dim).to(dtype)


class VesuviusModel(nn.Module):
    def __init__(self, z_dim=32, model_dim=128, num_layers=2, nhead=8):
        super().__init__()
        assert z_dim % 4 == 0, "z_dim must be divisible by 4 (two stride-2 stages)"
        self.z_dim = z_dim
        self.model_dim = model_dim
        z_out = z_dim // 4

        # 1. Stem: full-Z, half-resolution in H/W
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.skip_proj = nn.Conv2d(32, 32, 1)  # applied to mean-over-Z stem features (H/2)

        # 2. 3D residual encoder: (Z, H/2, W/2) -> (Z/4, H/8, W/8)
        self.encoder3d = nn.Sequential(
            BasicBlock3D(32, 64, stride=(2, 2, 2)),
            BasicBlock3D(64, 128, stride=(2, 2, 2)),
        )

        # 3. 3D -> 2D neck: fold the remaining Z into channels
        self.neck = nn.Conv2d(128 * z_out, model_dim, kernel_size=1)

        # 4. Transformer over (H/8 * W/8) tokens
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=model_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

        # 5. Decoder: H/8 -> H/4 -> H/2 (+skip) -> H
        self.up1 = nn.Sequential(nn.Conv2d(model_dim, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        b, c, z, h, w = x.shape
        assert z == self.z_dim, f"expected {self.z_dim} slices, got {z}"
        assert h % 8 == 0 and w % 8 == 0, "H and W must be divisible by 8"

        s = self.stem(x)                          # (B, 32, Z, H/2, W/2)
        skip = self.skip_proj(s.mean(dim=2))      # (B, 32, H/2, W/2)

        f = self.encoder3d(s)                     # (B, 128, Z/4, H/8, W/8)
        f = f.reshape(b, -1, f.shape[-2], f.shape[-1])
        f = self.neck(f)                          # (B, D, H/8, W/8)

        hh, ww = f.shape[-2:]
        tokens = f.flatten(2).transpose(1, 2)     # (B, hh*ww, D)
        tokens = tokens + sincos_2d(hh, ww, self.model_dim, tokens.device, tokens.dtype)
        tokens = self.transformer(tokens)
        f = tokens.transpose(1, 2).reshape(b, self.model_dim, hh, ww)

        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)          # H/4
        f = self.up1(f)
        f = F.interpolate(f, size=skip.shape[-2:], mode="bilinear", align_corners=False)    # H/2
        f = self.up2(torch.cat([f, skip], dim=1))
        f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)             # H
        return self.head(f)


def dice_loss(logits, target, mask=None, smooth=1.0):
    """Soft Dice loss on logits (sigmoid applied internally); optional validity mask."""
    probs = torch.sigmoid(logits)
    if mask is not None:
        probs, target = probs * mask, target * mask
    probs, target = probs.reshape(-1), target.reshape(-1)
    inter = (probs * target).sum()
    return 1 - (2.0 * inter + smooth) / (probs.sum() + target.sum() + smooth)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesuviusModel().to(device)
    dummy = torch.randn(1, 1, 32, 256, 256, device=device)
    out = model(dummy)
    print(f"input {tuple(dummy.shape)} -> output {tuple(out.shape)} | params: {count_parameters(model)/1e6:.2f}M")
