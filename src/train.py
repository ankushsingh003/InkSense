




import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from src.dataset import create_dataloaders
from src.model import load_model, count_parameters
from src.utils import save_checkpoint, EarlyStopping, plot_training_history


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, prices) in enumerate(pbar):
        images = images.to(device)
        prices = prices.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler and config.USE_AMP:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, prices)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(prices.detach().cpu().numpy())
        
        # Update progress bar
        if batch_idx % config.LOG_INTERVAL == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    
    # Calculate MAE
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    return epoch_loss, mae


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, prices in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            prices = prices.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, prices)
            
            running_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(prices.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    
    # Calculate metrics
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    # R² score
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return epoch_loss, mae, rmse, r2


def train(resume_from=None):
    """
    Main training function
    
    Args:
        resume_from: Optional path to checkpoint to resume from
    """
    print("="*70)
    print("IMAGE VALUE PREDICTION - TRAINING")
    print("="*70)
    
    # Setup device
    device = torch.device(config.DEVICE)
    print(f"\n✓ Using device: {device}")
    
    # Create dataloaders
    print("\n📊 Loading datasets...")
    train_loader, val_loader, _ = create_dataloaders(
        config.TRAIN_CSV,
        config.VAL_CSV,
        config.TEST_CSV
    )
    
    # Create model
    print("\n🧠 Initializing model...")
    model = load_model(checkpoint_path=resume_from, device=device)
    count_parameters(model)
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Optimizer
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    
    # Learning rate scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config.SCHEDULER_PATIENCE,
                factor=config.SCHEDULER_FACTOR,
                verbose=True
            )
        elif config.SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.STEP_SIZE,
                gamma=config.GAMMA
            )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting training for {config.NUM_EPOCHS} epochs...")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n Epoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_mae, val_rmse, val_r2 = validate(
            model, val_loader, criterion, device
        )
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.4f}")
        print(f"  Val RMSE:   {val_rmse:.4f} | Val R²:    {val_r2:.4f}")
        
        # Learning rate scheduler step
        if scheduler:
            if config.SCHEDULER_TYPE == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config.BEST_MODEL_PATH
            )
            print(f" New best model saved!")
        
        # Save periodic checkpoint
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = config.CHECKPOINT_PATH.format(epoch)
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoint_path
            )
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\n⏹️ Early stopping triggered!")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model: {config.BEST_MODEL_PATH}")
    print("="*70)
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    # Run training
    model, train_losses, val_losses = train()
