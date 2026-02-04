# AeroSeg: Autonomous UAV Landing Zone & Obstacle Identification System

<div align="center">

**Real-time semantic segmentation for intelligent UAV flight systems**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Examples

| MobileNetV3 (Pre-trained) | Custom U-Net  |
|:---:|:---:|
| ![MobileNetV3 Example](assets/example_1.png) | ![U-Net Example](assets/example_unet_1.png) |
| *Identifies safe zones with high confidence* | *Refined mask boundaries for complex scenes* |

---

## Overview

AeroSeg is a computer vision pipeline designed for **autonomous UAV landing zone identification**. It uses semantic segmentation to classify aerial imagery into three categories:

| Category | Color | Description |
|----------|-------|-------------|
| **Safe** | 🟢 Green | Suitable landing surfaces (grass, pavement, open ground) |
| **Hazard** | 🔴 Red | Obstacles to avoid (buildings, vehicles, people) |
| **Water** | 🔵 Blue | Water bodies (rivers, lakes, pools) |

### Key Features

- ⚡ **Low-latency inference** using MobileNetV3 backbone or **Custom U-Net**
- 🧠 **Custom U-Net Architecture** with attention gates for precise segmentation
- 🎯 **Central ROI analysis** simulating UAV downward camera for landing zone focus
- 📈 **Complete Training Pipeline** for Semantic Drone Dataset
- 📊 **Safety scoring system** with configurable thresholds
- 🖼️ Supports both **image and video** processing
- 🚤 **Memory-efficient inference** with automatic resizing to prevent OOM
- 🔧 **Modular architecture** for easy integration and extension

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AeroSeg Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Input Image │───▶│ AeroSegModel │───▶│ImageProcessor │  │
│  │   (RGB)     │    │  (DeepLabV3) │    │  (ROI + Viz)  │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│                            │                    │           │
│                            ▼                    ▼           │
│                    ┌──────────────┐    ┌───────────────┐   │
│                    │Category Mask │    │ Safety Score  │   │
│                    │ (H×W) 0/1/2  │    │ + Status      │   │
│                    └──────────────┘    └───────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Model Options

- **MobileNetV3-Large**: (Default) Optimized for mobile/edge deployment, pre-trained on COCO.
- **Custom U-Net**: (~24M parameters) Built from scratch with attention gates for high-accuracy segmentation.
- **Light U-Net**: (~5M parameters) Lightweight variant designed for extreme edge devices.

### Class Mapping
The model maps complex indices to three simplified UAV categories:
1. **SAFE**: Suitable landing surfaces.
2. **HAZARD**: Obstacles (buildings, cars, trees, etc.).
3. **WATER**: Water bodies.

---

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

```bash
# Clone or navigate to project
cd AI-Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Pre-trained models |
| opencv-python | ≥4.8.0 | Image/video processing |
| matplotlib | ≥3.7.0 | Visualization |
| numpy | ≥1.24.0 | Array operations |

---

## Usage

### Process Single Image

```bash
python main.py --image aerial_view.jpg
```

### Save Output

```bash
python main.py --image aerial_view.jpg --output result.png
```

### Run with Custom U-Net
```bash
python main.py --image aerial_view.jpg --model unet --checkpoint checkpoints_unet20/best.pth
```

### Process Video
```bash
python main.py --video flight_footage.mp4 --output processed.mp4
```

### Advanced Options

```bash
# Custom ROI size (default: 200x200)
python main.py --image aerial.jpg --roi-size 300

# Force CPU inference
python main.py --image aerial.jpg --device cpu

# Headless mode (no display)
python main.py --image aerial.jpg --output result.png --no-display
```

---

## Output

### Terminal Output

```
============================================================
LANDING ZONE ANALYSIS RESULTS
============================================================
  Safe Area:     72.45%
  Hazard Area:   18.32%
  Water Area:     9.23%
  Safety Score:  63.47
------------------------------------------------------------
  Inference:     48.23ms
============================================================

  >>> [SAFETY STATUS: SECURE] <<<

============================================================
```

### Visualization

The output image shows:
- **Green overlay**: Safe landing areas
- **Red overlay**: Hazards/obstacles
- **Blue overlay**: Water bodies
- **ROI box**: Central region analyzed for landing (green = safe, red = hazard)

---

## Project Structure

```
AI-Project/
├── main.py           # CLI entry point
├── model.py          # AeroSegModel class (Inference wrapper)
├── custom_model.py   # Custom U-Net architectures (UNet, LightUNet)
├── train.py          # Training pipeline & dataset handling
├── processor.py      # ImageProcessor class (ROI + visualization)
├── download_data.py  # Script to download Semantic Drone Dataset
├── organize_data.py  # Utility to structure the dataset
├── requirements.txt  # Project dependencies
└── README.md         # Documentation
```

---

## Training

The project includes a full training pipeline for the **Semantic Drone Dataset**.

```bash
# 1. Download data
python download_data.py

# 2. Organize data
python organize_data.py --data ./data

# 3. Start training (Custom U-Net)
python train.py --data ./data --model unet --epochs 50
```

---

## Technical Notes

### Low-Latency Design

This system is optimized for **real-time inference** on resource-constrained platforms:

1. **MobileNetV3 backbone**: 5.4M parameters (vs 60M+ for ResNet-101)
2. **Single forward pass**: No post-processing NMS or similar operations
3. **Efficient inference**: ~50ms on GPU, ~200ms on CPU

### Safety Score Calculation

```python
safety_score = safe_percent - (hazard_percent × 0.5) - (water_percent × 0.3)
```

A landing zone is marked **SECURE** when:
- Safe area ≥ 70%
- Hazard area < 20%

---

## Future Enhancements

### Phase 2: Domain-Specific Training

For production deployment, fine-tune on aerial datasets:

| Dataset | Description |
|---------|-------------|
| [Semantic Drone Dataset](https://www.kaggle.com/datasets/santurini/semantic-drone-dataset) | 400 urban aerial images, 20 classes |
| [UAVid](https://uavid.nl/) | Urban scene understanding, 8 classes |
| [AeroScapes](https://github.com/ishann/aeroscapes) | Aerial scene parsing |

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for intelligent flight systems** 🚁

</div>
