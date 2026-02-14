


import torch
import argparse
import pandas as pd
from PIL import Image

import config
from src.model import load_model
from src.dataset import get_transforms
from src.utils import load_image, reverse_price_transform, format_price



def predict_single_image(image_path_or_url, model, device):
    
    try:
        # Load and transform image
        image = load_image(image_path_or_url)
        if image is None:
            raise ValueError(f"Could not load image: {image_path_or_url}")
        
        transform = get_transforms(is_training=False)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            predicted_price = reverse_price_transform(output.item())
        
        return predicted_price
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def predict_batch(image_paths, model, device, batch_size=32):
    
    predictions = []
    transform = get_transforms(is_training=False)
    model.eval()
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load and transform images
        for path in batch_paths:
            try:
                image = load_image(path)
                if image:
                    batch_images.append(transform(image))
                else:
                    predictions.append(None)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                predictions.append(None)
        
        if batch_images:
            # Predict batch
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)
                predictions.extend([reverse_price_transform(o.item()) for o in outputs])
    
    return predictions


def main():
    
    parser = argparse.ArgumentParser(description='Predict product price from image')
    parser.add_argument('--image', type=str, help='Path to image file or URL')
    parser.add_argument('--batch', type=str, help='Text file with list of image paths')
    parser.add_argument('--model', type=str, default=config.BEST_MODEL_PATH, help='Model checkpoint path')
    parser.add_argument('--output', type=str, help='Output CSV file for batch predictions')
    args = parser.parse_args()
    
    # Load model
    device = torch.device(config.DEVICE)
    print(f"Loading model from {args.model}...")
    model = load_model(checkpoint_path=args.model, device=device)
    print("Model loaded successfully!")
    
    # Single image prediction
    if args.image:
        print(f"\nProcessing image: {args.image}")
        price = predict_single_image(args.image, model, device)
        
        if price:
            print(f"\n{'='*50}")
            print(f"Predicted Price: {format_price(price)}")
            print(f"{'='*50}")
        else:
            print("Prediction failed!")
    
    # Batch prediction
    elif args.batch:
        print(f"\nLoading image paths from {args.batch}...")
        with open(args.batch, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(image_paths)} images")
        print("\nRunning batch prediction...")
        predictions = predict_batch(image_paths, model, device)
        
        # Display results
        print(f"\n{'='*70}")
        print("PREDICTIONS:")
        print(f"{'='*70}")
        for path, price in zip(image_paths, predictions):
            status = format_price(price) if price else "ERROR"
            print(f"{path:50s} -> {status}")
        
        # Save to CSV
        if args.output:
            pd.DataFrame({
                'image_path': image_paths,
                'predicted_price': predictions
            }).to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    else:
        print("Please specify either --image or --batch")
        parser.print_help()


if __name__ == "__main__":
    main()
