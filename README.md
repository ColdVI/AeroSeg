# AeroSeg: Autonomous UAV Landing Zone & Obstacle Identification System

<div align="center">

**Real-time semantic segmentation for intelligent UAV flight systems**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

AeroSeg is a computer vision pipeline designed for **autonomous UAV landing zone identification**. It uses semantic segmentation to classify aerial imagery into three categories:

| Category | Color | Description |
|----------|-------|-------------|
| **Safe** | 🟢 Green | Suitable landing surfaces (grass, pavement, open ground) |
| **Hazard** | 🔴 Red | Obstacles to avoid (buildings, vehicles, people) |
| **Water** | 🔵 Blue | Water bodies (rivers, lakes, pools) |

### Key Features

- ⚡ **Low-latency inference** using MobileNetV3 backbone (~50ms on GPU)
- 🎯 **Central ROI analysis** simulating UAV downward camera for landing zone focus
- 📊 **Safety scoring system** with configurable thresholds
- 🖼️ Supports both **image and video** processing
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

### Model Details

- **Backbone**: MobileNetV3-Large (optimized for mobile/edge deployment)
- **Head**: DeepLabV3 (atrous spatial pyramid pooling)
- **Pre-trained on**: COCO/VOC dataset (21 classes)
- **Class mapping**: COCO classes → Safe/Hazard/Water categories

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
├── model.py          # AeroSegModel class (DeepLabV3 wrapper)
├── processor.py      # ImageProcessor class (ROI + visualization)
├── requirements.txt  # Dependencies
└── README.md         # Documentation
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
