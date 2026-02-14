"""
Model architecture for price prediction
Supports ResNet and EfficientNet backbones
"""
import torch
import torch.nn as nn
from torchvision import models
import config


class PricePredictionModel(nn.Module):
    """
    Image-based price prediction model
    Uses pretrained CNN backbone with custom regression head
    """
    
    def __init__(self, backbone='resnet50', pretrained=True, freeze_backbone=False):
        super(PricePredictionModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
            
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ Backbone frozen: {backbone}")
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Single output (price)
        )
        
        print(f"✓ Model initialized: {backbone}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Feature dim: {num_features}")
        print(f"  - Output: 1 (regression)")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Tensor of shape (batch_size, 1) with predicted prices
        """
        # Extract features
        features = self.backbone(x)
        
        # Predict price
        price = self.regressor(features)
        
        return price.squeeze()  # Remove last dimension


def load_model(checkpoint_path=None, device=None):
    """
    Load model from checkpoint or create new model
    
    Args:
        checkpoint_path: Path to saved checkpoint (.pth file)
        device: Device to load model on
        
    Returns:
        model: Loaded model
    """
    if device is None:
        device = config.DEVICE
    
    # Create model
    model = PricePredictionModel(
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE
    )
    
    # Load checkpoint if provided
    if checkpoint_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print(f"✓ Model loaded from: {checkpoint_path}")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
            print("  Using randomly initialized model")
    
    model = model.to(device)
    return model


def count_parameters(model):
    """Count trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    return trainable_params


if __name__ == "__main__":
    print("Testing model architecture...")
    
    # Create model
    model = load_model()
    
    # Count parameters
    count_parameters(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, *config.IMAGE_SIZE)
    
    if config.DEVICE == 'cuda':
        dummy_input = dummy_input.cuda()
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Sample predictions: {output[:3].cpu().numpy()}")
