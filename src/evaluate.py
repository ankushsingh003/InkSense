

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

import config
from src.dataset import create_dataloaders
from src.model import load_model
from src.utils import reverse_price_transform, visualize_predictions


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_targets, all_images = [], [], []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, prices in tqdm(test_loader, desc="Testing"):
            images, prices = images.to(device), prices.to(device)
            outputs = model(images)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(prices.cpu().numpy())
            if len(all_images) < 20:
                all_images.extend(images.cpu())
    
    # Convert and reverse transformations
    preds, targets = np.array(all_preds), np.array(all_targets)
    if config.USE_LOG_TRANSFORM:
        preds_orig = np.array([reverse_price_transform(p) for p in preds])
        targets_orig = np.array([reverse_price_transform(t) for t in targets])
    else:
        preds_orig, targets_orig = preds, targets
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(targets_orig, preds_orig),
        'RMSE': np.sqrt(mean_squared_error(targets_orig, preds_orig)),
        'R2': r2_score(targets_orig, preds_orig),
        'MAPE': np.mean(np.abs((targets_orig - preds_orig) / targets_orig)) * 100,
        'Median_AE': np.median(np.abs(targets_orig - preds_orig))
    }
    
    return {
        'predictions': preds, 'targets': targets,
        'predictions_original': preds_orig, 'targets_original': targets_orig,
        'images': all_images[:20], 'metrics': metrics
    }


def print_metrics(metrics):
    """Print evaluation metrics"""
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    for key, value in metrics.items():
        label = key.replace('_', ' ')
        if key in ['MAE', 'RMSE', 'Median_AE']:
            print(f"  {label:30s} ${value:.2f}")
        elif key == 'MAPE':
            print(f"  {label:30s} {value:.2f}%")
        else:
            print(f"  {label:30s} {value:.4f}")
    print("="*70)


def plot_results(results, plot_type='predictions'):
    """Plot predictions or error distributions"""
    targets, preds = results['targets_original'], results['predictions_original']
    
    if plot_type == 'predictions':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(targets, preds, alpha=0.5, s=10)
        ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Price ($)'), ax1.set_ylabel('Predicted Price ($)')
        ax1.set_title('Predicted vs Actual Prices', fontweight='bold')
        ax1.legend(), ax1.grid(True, alpha=0.3)
        
        # Residual plot
        residuals = targets - preds
        ax2.scatter(targets, residuals, alpha=0.5, s=10)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Actual Price ($)'), ax2.set_ylabel('Residual ($)')
        ax2.set_title('Residual Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    else:  # error distribution
        errors = np.abs(targets - preds)
        pct_errors = (errors / targets) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Absolute errors
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: ${np.mean(errors):.2f}')
        ax1.axvline(np.median(errors), color='g', linestyle='--', lw=2, label=f'Median: ${np.median(errors):.2f}')
        ax1.set_xlabel('Absolute Error ($)'), ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Absolute Errors', fontweight='bold')
        ax1.legend(), ax1.grid(True, alpha=0.3)
        
        # Percentage errors
        ax2.hist(pct_errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(pct_errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(pct_errors):.1f}%')
        ax2.axvline(np.median(pct_errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(pct_errors):.1f}%')
        ax2.set_xlabel('Percentage Error (%)'), ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Percentage Errors', fontweight='bold')
        ax2.legend(), ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_predictions(results, output_path):
    """Save predictions to CSV"""
    df = pd.DataFrame({
        'actual_price': results['targets_original'],
        'predicted_price': results['predictions_original'],
        'absolute_error': np.abs(results['targets_original'] - results['predictions_original']),
        'percentage_error': np.abs((results['targets_original'] - results['predictions_original']) 
                                   / results['targets_original']) * 100
    })
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    """Main evaluation function"""
    print("="*70)
    print("IMAGE VALUE PREDICTION - EVALUATION")
    print("="*70)
    
    device = torch.device(config.DEVICE)
    print(f"\nUsing device: {device}")
    
    # Load data and model
    print("\nLoading test dataset...")
    _, _, test_loader = create_dataloaders(config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)
    if test_loader is None:
        print("No test data found!")
        return
    
    print(f"\nLoading model from {config.BEST_MODEL_PATH}...")
    model = load_model(checkpoint_path=config.BEST_MODEL_PATH, device=device)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    print_metrics(results['metrics'])
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_results(results, 'predictions')
    plot_results(results, 'errors')
    
    if results['images']:
        visualize_predictions(
            results['images'][:4],
            torch.tensor(results['targets'][:4]),
            torch.tensor(results['predictions'][:4]),
            num_samples=4
        )
        plt.show()
    
    # Save results
    save_predictions(results, os.path.join(config.DATA_DIR, 'test_predictions.csv'))
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
