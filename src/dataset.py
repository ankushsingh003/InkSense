
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


class ImagePriceDataset(Dataset):
    
    def __init__(self, csv_path, transform=None, is_training=True):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.is_training = is_training
        
        # Filter out invalid prices
        self.data = self.data[
            (self.data['price'] >= config.MIN_PRICE) & 
            (self.data['price'] <= config.MAX_PRICE)
        ].reset_index(drop=True)
        
        print(f" Loaded {len(self.data)} samples from {csv_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (C, H, W)
            price: Float tensor (log-transformed if configured)
        """
        row = self.data.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', config.IMAGE_SIZE, color='white')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get price
        price = float(row['price'])
        
        # Apply log transform if configured
        if config.USE_LOG_TRANSFORM:
            price = np.log1p(price)  # log(1 + price) to handle prices near 0
        
        return image, torch.tensor(price, dtype=torch.float32)


def get_transforms(is_training=True):
    """
    Get image transformations
    
    Args:
        is_training: If True, apply data augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training and config.TRAIN_AUGMENT:
        transform_list = [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomRotation(config.RANDOM_ROTATION),
            transforms.RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP),
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER['brightness'],
                contrast=config.COLOR_JITTER['contrast'],
                saturation=config.COLOR_JITTER['saturation'],
                hue=config.COLOR_JITTER['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ]
    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders(train_csv, val_csv, test_csv=None):
    """
    Create DataLoaders for training, validation, and optionally test sets
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ImagePriceDataset(
        train_csv,
        transform=get_transforms(is_training=True),
        is_training=True
    )
    
    val_dataset = ImagePriceDataset(
        val_csv,
        transform=get_transforms(is_training=False),
        is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    test_loader = None
    if test_csv and os.path.exists(test_csv):
        test_dataset = ImagePriceDataset(
            test_csv,
            transform=get_transforms(is_training=False),
            is_training=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if config.DEVICE == 'cuda' else False
        )
    
    print(f"\n{'='*50}")
    print(f"DataLoaders Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")
    print(f"{'='*50}\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset module...")
    
    # Check if data files exist
    if os.path.exists(config.TRAIN_CSV):
        dataset = ImagePriceDataset(config.TRAIN_CSV, transform=get_transforms())
        print(f"✓ Dataset size: {len(dataset)}")
        
        # Test loading one sample
        img, price = dataset[0]
        print(f"✓ Sample image shape: {img.shape}")
        print(f"✓ Sample price: {price.item():.2f}")
    else:
        print(f"⚠ Data files not found. Run data preprocessing first.")
