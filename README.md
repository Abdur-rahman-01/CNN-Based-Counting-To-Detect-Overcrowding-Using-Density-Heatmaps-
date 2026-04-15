# 🚇 Metro Crowd Density Estimator

A PyTorch-based crowd counting system using **CSRNet** (Convolutional Neural Networks for Understanding Highly Congested Scenes) deployed as a Streamlit web application for Indian metro stations.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Why This Project Matters](#why-this-project-matters)
3. [The Journey: From Failed Approaches to Success](#the-journey-from-failed-approaches-to-success)
4. [Model Architecture](#model-architecture)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results & Performance](#results--performance)
9. [File Structure](#file-structure)
10. [References](#references)
11. [License](#license)

---

## 🎯 Project Overview

This project implements a computer vision solution for estimating crowd density in metro/public transport environments. It uses CSRNet architecture with dilated convolutions to generate density maps and estimate people count in highly congested scenes.

### Key Features

- **CSRNet Model**: State-of-the-art crowd counting using VGG16 frontend + dilated convolutional backend
- **Web Interface**: Streamlit-based UI for real-time crowd density analysis
- **Multi-approach Comparison**: Includes documentation of alternative approaches that failed
- **Indian Metro Domain**: Custom-trained on Indian metro station images

---

## 🔬 Why This Project Matters

Crowd monitoring in public transport is critical for:
- **Safety**: Preventing stampedes and managing emergency situations
- **Efficiency**: Optimizing crowd flow and resource allocation
- **Planning**: Data-driven decisions for infrastructure development

Traditional methods like manual counting are error-prone and time-consuming. This AI solution provides real-time, accurate crowd density estimation.

---

## 🛠️ The Journey: From Failed Approaches to Success

This project documents three distinct approaches tried before achieving success with CSRNet:

### Approach 1: CNN Classifier (ResNet18) — ❌ FAILED

**Method**: Treated crowd counting as a 3-class classification problem
- Low: 0-30 people
- Medium: 31-80 people
- High: 81+ people

**Why It Failed**:
- **Coarse Classification**: Only provides categorical labels, not exact counts
- **No Spatial Understanding**: Cannot generate density maps showing where crowds concentrate
- **MAE ~55**: Poor precision for precise counting needs
- **Unsuitable for Zone-level Detection**: Cannot identify specific overcrowded areas

**Code**: [`approach1_classifier/classifier.py`](approach1_classifier/classifier.py)

---

### Approach 2: YOLOv8 Object Detection — ❌ FAILED

**Method**: Used pretrained YOLOv8n to detect individual people as bounding boxes

**Why It Failed**:
- **Severe Undercounting**: MAE of 283.23 due to heavy occlusion in dense crowds
- **Occlusion Sensitivity**: When crowds are dense, people overlap and detectors fail
- **Fundamentally Unsuitable**: Object detection assumes distinguishable objects; crowds are not objects
- **Under-detection Rate**: 50%+ of images showed YOLO predicting less than 50% of actual count

**Code**: [`approach2_yolo/aproach2_yolo.py`](approach2_yolo/aproach2_yolo.py)

---

### Approach 3: CSRNet with Dilated Convolutions — ✅ SUCCESS

**Method**: Density map regression using CSRNet architecture

**Why It Succeeded**:
- **Density Maps**: Outputs spatial density maps showing crowd distribution
- **Handles Occlusion**: Doesn't rely on detecting individual people
- **End-to-end Regression**: Directly predicts count from image
- **Dilated Convolutions**: Expanded receptive field captures global context
- **MAE: 12.36** — Dramatically better than all other approaches

**Key Insight**: Crowd counting is fundamentally a regression problem on density maps, not object detection or classification.

---

## 🏗️ Model Architecture

CSRNet consists of two main components:

### Frontend (VGG16-based)
```
Input → 64 → 64 → M → 128 → 128 → M → 256 → 256 → 256 → M → 512 → 512 → 512
```
- First 10 convolutional layers from VGG16 (pretrained on ImageNet)
- Extracts low-level features

### Backend (Dilated Convolutions)
```
512 → 512 → 512 → 256 → 128 → 64 (dilation=2)
```
- Dilated convolutions with rate=2 expand receptive field
- Captures global context without losing resolution
- Produces 64×64 density map

### Technical Details

| Component | Specification |
|-----------|---------------|
| Frontend | VGG16 first 10 conv layers |
| Backend | 6 dilated conv layers (dilation=2) |
| Output | 64×64 density map |
| Loss | MSE (Mean Squared Error) |
| Optimizer | SGD (lr=1e-7, momentum=0.95) |

---

## 📊 Dataset

### Training Data

| Source | Images | Description |
|--------|--------|-------------|
| ShanghaiTech Part A | 300 | Highly congested scenes |
| ShanghaiTech Part B | 400 | Medium-density scenes |
| Indian Metro | 88 | Custom metro station images |
| **Total** | **788** | Combined training set |

### Validation Data
- Indian metro station images in `my_data/val/`

### Format
- **Images**: JPG/PNG
- **Ground Truth**: NumPy (.npy) density maps
- **Density Map Generation**: Points annotations converted using Gaussian kernels (σ=15)

---

## 🛠️ Installation

```bash
# Navigate to project directory
cd CSRNet-pytorch

# Install dependencies
pip install torch torchvision
pip install opencv-python pillow numpy h5py
pip install streamlit
pip install ultralytics  # For YOLOv8 approach
pip install matplotlib
```

---

## 🚀 Usage

### Training

```bash
python train.py train_json val_json GPU_ID TASK_ID
```

Example:
```bash
python train.py part_A_train.json part_A_val.json 0 0
```

### Inference (Python Script)

```bash
python inference.py
```

### Streamlit Web Application

```bash
cd CSRNet-pytorch
streamlit run streamlit_app.py
```

The app provides:
- Image upload interface
- Real-time crowd density estimation
- Density map visualization
- Heatmap overlay
- Status classification (Normal/Moderate/Crowded/Overcrowded)

---

## 📈 Results & Performance

### Model Comparison

| Approach | MAE | MSE | Status |
|----------|-----|-----|--------|
| CNN Classifier | ~55 | — | ❌ Failed |
| YOLOv8 | 283.23 | — | ❌ Failed |
| **CSRNet v3** | **12.36** | **126.44** | ✅ Success |

### Alert Thresholds

| Status | People Count | Action |
|--------|--------------|--------|
| 🟢 Normal | 0-29 | No action needed |
| 🟡 Moderate | 30-69 | Monitor situation |
| 🟠 Crowded | 70-119 | Consider crowd management |
| 🔴 Overcrowded | 120+ | Immediate action required |

### Training Progress

- **Epochs**: 50
- **Best MAE**: 12.36 (achieved at epoch 27)
- **Final MSE**: 126.44

---

## 📁 File Structure

```
CSRNet-pytorch/
├── model.py                      # CSRNet model architecture
├── train.py                      # Training script
├── dataset.py                    # Dataset loader
├── image.py                      # Image loading utilities
├── utils.py                      # Model save/load utilities
├── inference.py                  # Standalone inference script
├── streamlit_app.py              # Streamlit web application
├── approach1_classifier/         # Failed CNN classifier approach
│   ├── classifier.py
│   └── classifier_viz.py
├── approach2_yolo/               # Failed YOLOv8 approach
│   ├── aproach2_yolo.py
│   └── yolo_demo.py
├── my_data/                      # Training/validation data
│   ├── train/
│   └── val/
├── results/                      # Output results & comparisons
├── weights/                      # Trained model weights
│   └── csrnet_v3_best.pth
├── part_A_train.json             # ShanghaiTech Part A train
├── part_A_val.json               # ShanghaiTech Part A val
├── part_B_train.json             # ShanghaiTech Part B train
├── part_B_val.json               # ShanghaiTech Part B val
└── README.md                     # This file
```

---

## 📚 References

1. **CSRNet Paper**: Li, Y., Zhang, X., & Chen, D. (2018). *CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes*. CVPR 2018.

2. **ShanghaiTech Dataset**: Zhang, Y., et al. (2016). *Single-image crowd counting via multi-column convolutional neural network*. CVPR 2016.

3. **VGG16**: Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015.

---

## 📝 License

This is a student project for educational purposes.

---

## 👥 Authors

- **Student**: Abdur Rahman Qasim
- **Guide**: Dr. Shivani Yadao
- **Project**: AI/Computer Vision Project #14
- **Institution**: Indian Institute of Technology
- **Deadline**: April 15, 2026

---

## 🙏 Acknowledgments

- ShanghaiTech Dataset providers
- PyTorch community
- CSRNet original authors