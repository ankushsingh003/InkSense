<div align="center">

<img src="https://raw.githubusercontent.com/Ankushsingh003/Ink-Alchemist/main/image_for_eval/ir.png" alt="InkSense Banner" width="700"/>

# 🔬 InkSense
### AI-Powered 3D Ink Detection & Virtual Unrolling Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![WandB](https://img.shields.io/badge/WandB-Tracking-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)

*Detect, quantify, and map ancient ink from 3D X-ray CT scans of carbonized papyrus fragments — in milliseconds.*

[**🔬 Live Demo**](#-web-interface) · [**🚀 Quick Start**](#-quick-start) · [**🏗️ Architecture**](#-system-architecture) · [**📊 Web Report Flow**](#-web-report-generation-flow)

</div>

---

## 📜 Overview

**InkSense** is an industry-grade AI pipeline — the quest to digitally read thousands of carbonized Greek scrolls buried by Mount Vesuvius in 79 AD.

The pipeline uses a **Hybrid 3D-CNN + Transformer** model to detect the chemical signature of ink residue inside 3D X-ray CT volumes, without physically unrolling or damaging the fragile scrolls. Results are served through a premium web interface that shows ink coverage, heatmaps, and region analysis in real-time.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A["🗂️ Raw CT Scan Data\n3D X-ray Volumes (20GB+)"] --> B

    subgraph DataEng["📦 Phase 1 — Data Engineering"]
        B["Memory-Mapped\nNumPy Loader\n(mmap_mode='r')"]
        B --> C["Surface Extractor\n(Middle 16–32 Z-Slices)"]
        C --> D["Dynamic Tiler\n(256×256, stride=128)"]
    end

    D --> E

    subgraph Model["🧠 Phase 2 — Hybrid AI Model"]
        E["3D-ResNet Backbone\n(Volumetric Feature Extraction)"]
        E --> F["Transformer Encoder\n(Global Semantic Context)"]
        F --> G["2D Segmentation Head\n(Pixel-Level Mask Output)"]
    end

    G --> H

    subgraph MLOps["⚙️ Phase 3 — MLOps Pipeline"]
        H["WandB Experiment Tracker\n(Loss, Dice, Epoch Metrics)"]
        H --> I["Threshold Sweeper\n(Best F0.5 / Dice score)"]
        I --> J["Morphological Denoiser\n(MORPH_OPEN + MORPH_CLOSE)"]
    end

    J --> K["🖥️ InkSense Web Interface\n(Heatmap + Report)"]

    style DataEng fill:#1e1b4b,stroke:#6366f1,color:#c7d2fe
    style Model fill:#14532d,stroke:#10b981,color:#a7f3d0
    style MLOps fill:#7c2d12,stroke:#f59e0b,color:#fef3c7
    style K fill:#1e293b,stroke:#6366f1,color:#e2e8f0
```

---

## 🌐 Web Report Generation Flow

> How InkSense transforms a user-uploaded image into a complete ink analysis report.

```mermaid
sequenceDiagram
    actor User
    participant UI as 🖥️ Browser UI
    participant Engine as ⚙️ JS Analysis Engine
    participant Canvas as 🎨 HTML Canvas
    participant Report as 📊 Report Panel

    User->>UI: Upload Image (PNG/JPG/TIF)
    UI->>UI: Detect file format
    alt TIF/TIFF
        UI->>Engine: Decode via UTIF.js → RGBA pixels
    else Standard format
        UI->>Engine: FileReader → DataURL → Image element
    end
    Engine->>Canvas: Draw decoded pixels to sourceCanvas
    UI->>UI: Show image preview ✅
    User->>UI: Click "Run Ink Analysis"
    UI->>Engine: Start processing steps (animated)
    loop 8 Processing Steps
        Engine->>UI: Update step label + progress bar
    end
    Engine->>Canvas: getImageData() → pixel array
    Engine->>Engine: Compute luminance & saturation per pixel
    Engine->>Engine: Build inkMap[] (0.0–1.0 strength per pixel)
    Engine->>Engine: Count ink pixels & coverage %
    Engine->>Engine: estimateRegions() → BFS connected components
    Engine->>Canvas: Draw dimmed original + heatmap overlay
    Engine->>Canvas: Draw bounding box around ink region
    Engine->>Report: Render metrics (Coverage %, Ink px, Regions, Latency)
    Engine->>Report: Animate coverage bar
    Engine->>Report: Write plain-English summary
    Report->>User: Full ink report displayed ✅
    User->>UI: Click "Download Ink Mask"
    UI->>User: canvas.toDataURL() → PNG download
```

---

## 🛠️ Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data** | NumPy `mmap_mode='r'` | Handle 20GB+ CT volumes without RAM overflow |
| **Model** | PyTorch 3D-ResNet + Transformer | Volumetric feature extraction + semantic mapping |
| **Training** | WandB + BCE + Dice Loss | Experiment tracking and hybrid loss optimisation |
| **Post-Processing** | OpenCV Morphology | Threshold sweeping + noise denoising |
| **Web Frontend** | Vite + Vanilla JS | Ultra-fast, zero-framework interactive UI |
| **TIF Decoding** | UTIF.js | Client-side TIFF byte-level decoder |
| **Web Server** | Python `http.server` | Static file serving of production build |
| **Container** | Docker Multi-Stage | Node → Python build pipeline |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Ankushsingh003/Ink-Alchemist.git
cd Ink-Alchemist
pip install -r requirements.txt
```

### 2. Prepare Data
```powershell
cd process_fragments
./download_fragment_1.ps1
./unzip_fragment_1.ps1
```

### 3. Run the AI Pipeline
```bash
# Preprocess CT scan volumes
python data_preprocessing.py --fragment 1

# Train the model (10 epochs, WandB logged)
python train.py

# Evaluate & optimise thresholds
python evaluation.py
```

### 4. Launch the Web Interface
```bash
# Serve the InkSense web app
python serve_app.py
# → Open http://localhost:3000
```

---

## 🏆 Industrial Impact

| Trait | Detail |
|---|---|
| 🛡️ **Noise Resilience** | Bypasses CT artifacts and papyrus noise — focuses on the ink's chemical signature |
| ⚡ **Resource Efficiency** | Handles 20GB+ volumes on consumer hardware via memory mapping + tiling |
| 🔬 **Domain Adaptability** | Techniques transfer to Medical AI (CT/MRI tumour segmentation) and Aerospace NDT |
| 🌍 **Cultural Impact** | Enables reading of thousands of lost ancient scrolls without physical damage |

---

## 🏗️ Project Structure

```
InkSense/
├── model.py                  # Hybrid 3D-CNN + Transformer architecture
├── train.py                  # Training loop with WandB integration
├── evaluation.py             # Post-processing & threshold optimisation
├── data_preprocessing.py     # Volume compression and tiling system
├── visualize_impact.py       # Presentation heatmap generator
├── serve_app.py              # Python web server (port 3000)
├── Dockerfile                # Multi-stage Docker build (Node → Python)
├── requirements.txt          # Python dependencies
├── ink-alchemist-web/        # Vite web application (InkSense UI)
│   ├── index.html            # Company website + analyser tool
│   ├── src/
│   │   ├── main.js           # Ink detection engine + UI logic
│   │   └── style.css         # Glassmorphism design system
│   └── dist/                 # Production build (served by serve_app.py)
├── process_fragments/        # Data acquisition PowerShell scripts
├── processed/                # Pre-processed .npy volumes
├── checkpoints/              # Saved model weights
└── results/                  # Output heatmaps and reports
```

---

## 🚀 Deployment (Render)

To deploy this project on Render:

1. **Push to GitHub**: Ensure all changes are pushed to your repository (`github.com/ankushsingh003/Ink-Alchemist`).
2. **Create Web Service**: In Render, click "New" -> "Web Service".
3. **Connect Repo**: Connect your GitHub repository.
4. **Configuration**:
   - **Environment**: `Docker`
   - **Plan**: Any plan (the optimized image is lightweight).
5. **Environment Variables**: Render automatically sets the `PORT` variable.
6. **Deploy**: Click "Create Web Service".

---

<div align="center">

Made with ❤️ · **InkSense** © 2025

</div>
