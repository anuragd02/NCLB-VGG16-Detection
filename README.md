# 🌽 NCLB Detection using VGG16 (Reproducible Deep Learning Pipeline)

This repository presents a reproducible deep learning framework for the detection of **Northern Corn Leaf Blight (NCLB)** using transfer learning with the VGG16 architecture.

The implementation includes:
- End-to-end training pipeline
- Data preprocessing and augmentation
- Model evaluation using multiple metrics
- Learning curve visualization
- Reproducibility controls (fixed random seed)

---

## 📌 Overview

Northern Corn Leaf Blight (NCLB) is a major disease affecting maize crops, leading to significant yield loss. Early and accurate detection is critical for effective disease management.

This work leverages **transfer learning with VGG16** to classify maize leaf images into:
- Healthy
- Diseased (NCLB)

---

## 🧠 Model Architecture

- Base Model: **VGG16 (pretrained on ImageNet)**
- Feature Extraction: Frozen convolutional layers
- Custom Head:
  - Global Average Pooling
  - Dense Layer (128 units, ReLU)
  - Dropout (0.5)
  - Output Layer (Sigmoid for binary classification)

---

## ⚙️ Hyperparameters

| Parameter        | Value        |
|------------------|-------------|
| Optimizer        | Adam        |
| Learning Rate    | 0.001       |
| Batch Size       | 32          |
| Epochs           | 10          |
| Dropout Rate     | 0.5         |
| Input Size       | 224 × 224   |
| Loss Function    | Binary Crossentropy |

---

## 🔁 Reproducibility

To ensure consistent results, a fixed random seed is used across:
- Python (`random`)
- NumPy
- TensorFlow

```python
SEED = 42
