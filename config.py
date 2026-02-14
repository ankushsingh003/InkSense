"""
Configuration file for Image Value Prediction Project
Contains all hyperparameters, paths, and model settings
"""
import os
import torch

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(BASE_DIR, "7GB_IMAGE_TO_TEXT_DATASET")
IMAGE_DIR = os.path.join(RAW_DATA_DIR, "train_val_images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

# Dataset files
ANNOT_FILE = os.path.join(RAW_DATA_DIR, "annot.parquet")
IMG_FILE = os.path.join(RAW_DATA_DIR, "img.parquet")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Model checkpoint paths
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint_epoch_{}.pth")

# ==================== MODEL PARAMETERS ====================
# Image settings
IMAGE_SIZE = (224, 224)  # ResNet standard input size
IMAGE_CHANNELS = 3

# Model architecture
BACKBONE = "resnet50"  # Options: resnet18, resnet34, resnet50, efficientnet_b0
PRETRAINED = True
FREEZE_BACKBONE = False  # Set to True to freeze pretrained layers initially

# Output
NUM_CLASSES = 1  # Regression task (single price value)

# ==================== TRAINING HYPERPARAMETERS ====================
BATCH_SIZE = 32
NUM_WORKERS = 4  # DataLoader workers
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Optimizer settings
OPTIMIZER = "adam"  # Options: adam, adamw, sgd
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9  # For SGD

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "reduce_on_plateau"  # Options: reduce_on_plateau, step, cosine
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
STEP_SIZE = 10  # For StepLR
GAMMA = 0.1  # For StepLR

# ==================== DATA AUGMENTATION ====================
# Training augmentations
TRAIN_AUGMENT = True
RANDOM_ROTATION = 15
RANDOM_HORIZONTAL_FLIP = 0.5
COLOR_JITTER = {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1
}

# Normalization (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ==================== DATA SPLIT ====================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ==================== DEVICE CONFIGURATION ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_DEVICE_ID = 0  # GPU device ID if multiple GPUs available

# Mixed precision training (for faster training on GPU)
USE_AMP = True if DEVICE == "cuda" else False

# ==================== LOGGING & CHECKPOINTING ====================
SAVE_EVERY_N_EPOCHS = 5
LOG_INTERVAL = 10  # Log every N batches
VERBOSE = True

# TensorBoard
USE_TENSORBOARD = False
TENSORBOARD_DIR = os.path.join(BASE_DIR, "runs")

# ==================== PRICE PROCESSING ====================
# Price transformation (helps with modeling)
USE_LOG_TRANSFORM = True  # Use log(price) for training
MIN_PRICE = 0.01  # Minimum valid price
MAX_PRICE = 10000.0  # Maximum valid price

# ==================== WEB APP CONFIGURATION ====================
# Streamlit
STREAMLIT_PORT = 8501
STREAMLIT_TITLE = "🏷️ Product Price Predictor"

# Flask
FLASK_PORT = 5000
FLASK_DEBUG = True
FLASK_HOST = "0.0.0.0"

# Upload settings
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ==================== UTILITIES ====================
def ensure_dirs():
    """Create necessary directories if they don't exist"""
    dirs = [DATA_DIR, MODEL_DIR, NOTEBOOK_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    print(f"✓ All directories created/verified")

def get_device_info():
    """Print device information"""
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(CUDA_DEVICE_ID)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Mixed Precision: {'Enabled' if USE_AMP else 'Disabled'}")
    return DEVICE

if __name__ == "__main__":
    ensure_dirs()
    get_device_info()
    print(f"\n✓ Configuration loaded successfully!")
    print(f"  - Backbone: {BACKBONE}")
    print(f"  - Image Size: {IMAGE_SIZE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
