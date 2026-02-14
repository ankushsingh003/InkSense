"""
Utility functions for the project
Includes image downloading, preprocessing, and visualization helpers
"""
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from io import BytesIO
import config


def download_image(url, timeout=10):
    """
    Download image from URL
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL.Image or None if download fails
    """
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    
    except Exception as e:
        print(f"⚠ Error downloading image from {url}: {e}")
        return None


def load_image(image_path_or_url):
    """
    Load image from local path or URL
    
    Args:
        image_path_or_url: Local file path or HTTP(S) URL
        
    Returns:
        PIL.Image
    """
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        return download_image(image_path_or_url)
    else:
        return Image.open(image_path_or_url).convert('RGB')


def denormalize_image(tensor, mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD):
    """
    Denormalize a tensor image for visualization
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        numpy array suitable for visualization
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clip to valid range and convert to numpy
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    return img


def reverse_price_transform(log_price):
    """
    Reverse the log transformation applied to prices
    
    Args:
        log_price: Log-transformed price value
        
    Returns:
        Original price value
    """
    if config.USE_LOG_TRANSFORM:
        return np.expm1(log_price)  # inverse of log1p
    else:
        return log_price


def visualize_predictions(images, true_prices, pred_prices, num_samples=4):
    """
    Visualize model predictions
    
    Args:
        images: Batch of image tensors
        true_prices: Ground truth prices
        pred_prices: Predicted prices
        num_samples: Number of samples to display
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[i])
        
        # Reverse price transformations
        true_price = reverse_price_transform(true_prices[i].item())
        pred_price = reverse_price_transform(pred_prices[i].item())
        
        error = abs(true_price - pred_price)
        error_pct = (error / true_price) * 100 if true_price > 0 else 0
        
        # Display
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"True: ${true_price:.2f}\n"
            f"Pred: ${pred_price:.2f}\n"
            f"Error: {error_pct:.1f}%",
            fontsize=10
        )
    
    plt.tight_layout()
    return fig


def plot_training_history(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def format_price(price):
    """
    Format price for display
    
    Args:
        price: Price value
        
    Returns:
        Formatted string
    """
    if price < 1:
        return f"${price:.2f}"
    elif price < 1000:
        return f"${price:.2f}"
    else:
        return f"${price:,.2f}"


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            'backbone': config.BACKBONE,
            'image_size': config.IMAGE_SIZE,
            'use_log_transform': config.USE_LOG_TRANSFORM
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        dict with checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


if __name__ == "__main__":
    print("Testing utils module...")
    
    # Test image download
    test_url = "https://m.media-amazon.com/images/I/71O1w9TrNWL._AC_SX679_.jpg"
    img = download_image(test_url)
    
    if img:
        print(f"✓ Image downloaded: {img.size}")
    else:
        print("⚠ Image download failed")
    
    # Test price formatting
    prices = [0.99, 9.99, 99.99, 999.99, 9999.99]
    print("\nPrice formatting:")
    for price in prices:
        print(f"  {price} -> {format_price(price)}")
