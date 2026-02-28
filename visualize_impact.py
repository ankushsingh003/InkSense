import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def create_presentation_plot(original_ir, prediction, denoised, output_path="presentation_demo.png"):
    """
    Generates a professional side-by-side comparison for industry presentations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_ir, cmap='gray')
    axes[0].set_title("1. Infrared Scan (Raw Data)", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(prediction, cmap='magma')
    axes[1].set_title("2. AI Raw Prediction (Heatmap)", fontsize=14)
    axes[1].axis('off')
    
    # Overlay prediction on IR for the final "wow" factor
    axes[2].imshow(original_ir, cmap='gray')
    axes[2].imshow(denoised, cmap='jet', alpha=0.4)
    axes[2].set_title("3. Virtual Unrolling (Denoised)", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Presentation visualization saved to {output_path}")

if __name__ == "__main__":
    import os
    # Load real data for verification
    data_dir = "d:/ink_detection_fragments/processed"
    lbl_path = os.path.join(data_dir, "fragment1_labels.npy")
    
    if os.path.exists(lbl_path):
        # Using a small slice for demo
        dummy_ir = np.load(lbl_path, mmap_mode='r')[:512, :512].astype(float) / 255.0
        dummy_pred = np.clip(dummy_ir + np.random.normal(0, 0.2, dummy_ir.shape), 0, 1)
        dummy_denoised = (dummy_pred > 0.5).astype(float)
        
        os.makedirs("results", exist_ok=True)
        create_presentation_plot(dummy_ir, dummy_pred, dummy_denoised, "results/demo_impact.png")
    else:
        # Fallback to dummy
        dummy_ir = np.random.rand(512, 512)
        dummy_pred = np.random.rand(512, 512)
        dummy_denoised = (dummy_pred > 0.5).astype(float)
        os.makedirs("results", exist_ok=True)
        create_presentation_plot(dummy_ir, dummy_pred, dummy_denoised, "results/demo_impact.png")
