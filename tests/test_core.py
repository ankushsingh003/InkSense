import torch
import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import VesuviusModel
from train import VesuviusDataset

def test_model_output_shape():
    """
    Test that the model accepts (B, C, Z, H, W) and returns (B, 1, H, W).
    """
    device = torch.device('cpu')
    model = VesuviusModel().to(device)
    
    # Batch size 2, 1 channel, 32 slices, 256x256 tiles
    dummy_input = torch.randn(2, 1, 32, 256, 256).to(device)
    output = model(dummy_input)
    
    assert output.shape == (2, 1, 256, 256), f"Wrong output shape: {output.shape}"

def test_dataset_tile_generation():
    """
    Test that the dataset correctly generates tiles of the requested size.
    """
    # Create dummy data files
    vol = np.random.rand(32, 512, 512).astype(np.float32)
    lbl = np.random.rand(512, 512).astype(np.float32)
    msk = np.ones((512, 512)).astype(np.uint8)
    
    np.save("test_vol.npy", vol)
    np.save("test_lbl.npy", lbl)
    np.save("test_msk.npy", msk)
    
    ds = VesuviusDataset("test_vol.npy", "test_lbl.npy", "test_msk.npy", tile_size=256)
    
    # Get first item
    image, label = ds[0]
    
    # Dataset returns (1, Z, H, W)
    assert image.shape == (1, 32, 256, 256), f"Wrong image tile shape: {image.shape}"
    assert label.shape == (1, 256, 256), f"Wrong label tile shape: {label.shape}"
    
    # Cleanup
    os.remove("test_vol.npy")
    os.remove("test_lbl.npy")
    os.remove("test_msk.npy")

def test_model_gradients():
    """
    Ensure that gradients flow through the model back to inputs.
    """
    model = VesuviusModel()
    input_data = torch.randn(1, 1, 32, 128, 128, requires_grad=True)
    output = model(input_data)
    
    loss = output.mean()
    loss.backward()
    
    assert input_data.grad is not None, "Gradients are not flowing back to input!"
