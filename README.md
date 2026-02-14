# 🏷️ Image Value Prediction

An end-to-end deep learning project that predicts product prices from images using PyTorch and pretrained CNNs (ResNet/EfficientNet). Includes training pipeline, evaluation tools, and web applications for deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Applications](#web-applications)
- [API Documentation](#api-documentation)
- [Results](#results)
- [Contributing](#contributing)

---

## ✨ Features

- 🧠 **Deep Learning**: ResNet50/EfficientNet backbone with custom regression head
- 📊 **Complete Pipeline**: Data preprocessing, training, evaluation, and inference
- 🎨 **Modern Web UI**: Both Streamlit and Flask interfaces
- 🚀 **Production Ready**: RESTful API, error handling, and logging
- 📈 **Comprehensive Metrics**: MAE, RMSE, R², and visualization tools
- ⚡ **GPU Support**: CUDA acceleration with mixed precision training
- 🔄 **Data Augmentation**: Advanced transforms for robust training

---

## 📁 Project Structure

```
image-value-pred/
│
├── config.py                    # Configuration & hyperparameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── data/                        # Processed datasets
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── dataset.py              # Dataset & DataLoader
│   ├── model.py                # Model architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── predict.py              # Inference script
│   └── utils.py                # Helper functions
│
├── models/                      # Saved model checkpoints
│   └── best_model.pth
│
├── notebooks/                   # Jupyter notebooks
│   └── (EDA & experiments)
│
├── app.py                       # Streamlit web app
├── flask_app.py                 # Flask API
│
└── templates/                   # HTML templates
    └── index.html
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd image-value-pred
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python config.py
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

Create CSV files with the following structure:
```csv
image_path,price
path/to/image1.jpg,29.99
path/to/image2.jpg,149.50
```

Place them in the `data/` directory:
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

### 2. Train the Model
```bash
python src/train.py
```

### 3. Evaluate the Model
```bash
python src/evaluate.py
```

### 4. Run Web Application

**Streamlit:**
```bash
streamlit run app.py
```

**Flask:**
```bash
python flask_app.py
```

---

## 📖 Usage

### Training

```bash
python src/train.py
```

**Features:**
- Automatic checkpointing
- Early stopping
- Learning rate scheduling
- Mixed precision training (GPU)
- Training history visualization

**Configuration:**
Edit `config.py` to adjust:
- Model backbone (ResNet18/34/50, EfficientNet)
- Image size
- Batch size
- Learning rate
- Number of epochs
- Data augmentation settings

### Evaluation

```bash
python src/evaluate.py
```

**Outputs:**
- Comprehensive metrics (MAE, RMSE, R², MAPE)
- Prediction vs actual scatter plots
- Error distribution histograms
- Sample predictions visualization
- Results CSV export

### Prediction

**Single Image:**
```bash
python src/predict.py --image path/to/image.jpg
```

**From URL:**
```bash
python src/predict.py --image https://example.com/product.jpg
```

**Batch Prediction:**
```bash
python src/predict.py --batch image_list.txt --output predictions.csv
```

---

## 🌐 Web Applications

### Streamlit Interface

A user-friendly web application with:
- Image upload or URL input
- Real-time predictions
- Modern, responsive design
- Visual feedback

**Run:**
```bash
streamlit run app.py
```

**Access:** http://localhost:8501

### Flask API

RESTful API with modern web interface:

**Run:**
```bash
python flask_app.py
```

**Access:** http://localhost:5000

---

## 🔌 API Documentation

### Endpoints

#### `POST /predict`

Predict price from image.

**Request (JSON):**
```json
{
  "image_url": "https://example.com/product.jpg"
}
```

**Request (Form Data):**
```
file: <uploaded_image_file>
```

**Response:**
```json
{
  "success": true,
  "predicted_price": 99.99,
  "formatted_price": "$99.99",
  "currency": "USD",
  "model_trained": true
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "backbone": "resnet50"
}
```

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **MAE** | $X.XX |
| **RMSE** | $X.XX |
| **R² Score** | 0.XX |
| **MAPE** | X.X% |

*Results will be updated after training*

### Sample Predictions

*(Add screenshots or examples after training)*

---

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Model
BACKBONE = "resnet50"
IMAGE_SIZE = (224, 224)
PRETRAINED = True

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 🧪 Development

### Project Workflow

1. **Data Preparation**: Clean and split dataset
2. **EDA**: Explore data in Jupyter notebooks
3. **Training**: Train model with optimal hyperparameters
4. **Evaluation**: Assess performance on test set
5. **Deployment**: Deploy web applications
6. **Monitoring**: Track predictions and retrain as needed

### Testing

```bash
# Test dataset loading
python src/dataset.py

# Test model architecture
python src/model.py

# Test utilities
python src/utils.py
```

---

## 📝 Dataset Format

The dataset should have the following structure:

**CSV Format:**
```csv
image_path,price
train_val_images/img001.jpg,29.99
train_val_images/img002.jpg,149.50
```

**Requirements:**
- Images must be accessible at the specified paths
- Prices should be numeric (float)
- Supported formats: JPG, JPEG, PNG, WEBP

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `IMAGE_SIZE`
- Set `USE_AMP = False`

**2. Model not found**
- Ensure you've trained the model first
- Check `BEST_MODEL_PATH` in `config.py`

**3. Data loading errors**
- Verify CSV files exist in `data/` directory
- Check image paths are correct
- Ensure images are readable

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Pretrained models from torchvision
- Streamlit and Flask for web frameworks

---

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Made with ❤️ using PyTorch and Python**
