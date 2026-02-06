# 📚 Code-Theory Linkage Document

## Overview

This document connects every piece of code in this project with its theoretical foundations, mathematical derivations, and practical explanations. As a beginner-friendly guide, each section explains:

1. **What** - The concept being implemented
2. **Why** - The theoretical motivation
3. **How** - Mathematical foundations
4. **Code** - Link to actual implementation

---

## Table of Contents

- [1. Image Processing Fundamentals](#1-image-processing-fundamentals)
- [2. Neural Network Basics](#2-neural-network-basics)
- [3. Convolutional Neural Networks](#3-convolutional-neural-networks)
- [4. Object Detection Concepts](#4-object-detection-concepts)
- [5. YOLO Architecture](#5-yolo-architecture)
- [6. Loss Functions](#6-loss-functions)
- [7. Evaluation Metrics](#7-evaluation-metrics)
- [8. Training Optimization](#8-training-optimization)

---

## 1. Image Processing Fundamentals

### 1.1 Digital Images as Matrices

**Theory:**
A digital image is represented as a matrix of pixel values. For grayscale images, each pixel is a single intensity value (0-255). For RGB color images, each pixel has 3 values (Red, Green, Blue).

**Mathematics:**

For a grayscale image of size H×W:
```
I ∈ ℝ^(H×W)
I[i,j] ∈ [0, 255]  ∀ i ∈ [0,H-1], j ∈ [0,W-1]
```

For an RGB image of size H×W×3:
```
I ∈ ℝ^(H×W×3)
I[i,j,c] ∈ [0, 255]  ∀ c ∈ {R,G,B}
```

**Code Link:** `src/math_from_scratch/image_basics.py`

---

### 1.2 Image Normalization

**Theory:**
Neural networks work better with normalized inputs. We scale pixel values from [0, 255] to [0, 1] or [-1, 1].

**Mathematics:**

Min-Max Normalization:
```
x_normalized = (x - x_min) / (x_max - x_min)
```

For images (0-255 to 0-1):
```
I_normalized = I / 255.0
```

Z-score Normalization:
```
x_standardized = (x - μ) / σ
```

Where μ is mean and σ is standard deviation.

**Why it matters:**
- Prevents gradient explosion/vanishing
- Faster convergence during training
- Equal importance to all features

**Code Link:** `notebooks/02_data_preprocessing.ipynb`, Cell 3

---

## 2. Neural Network Basics

### 2.1 Perceptron (Single Neuron)

**Theory:**
The perceptron is the fundamental building block of neural networks. It takes inputs, applies weights, adds bias, and passes through an activation function.

**Mathematics:**

```
z = Σ(w_i × x_i) + b = W^T × X + b
a = σ(z)
```

Where:
- x_i: input features
- w_i: weights
- b: bias
- σ: activation function
- a: output (activation)

**Visual Representation:**
```
    x₁ ──w₁──┐
    x₂ ──w₂──┼──[Σ + b]──[σ]──► output
    x₃ ──w₃──┘
```

**Code Link:** `src/math_from_scratch/activations.py`

---

### 2.2 Activation Functions

**Theory:**
Activation functions introduce non-linearity, allowing neural networks to learn complex patterns.

**Common Activation Functions:**

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| Sigmoid | σ(x) = 1/(1+e^(-x)) | (0, 1) | Binary classification |
| Tanh | tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Hidden layers |
| ReLU | max(0, x) | [0, ∞) | Most common |
| Leaky ReLU | max(αx, x), α=0.01 | (-∞, ∞) | Prevents dead neurons |
| SiLU/Swish | x × sigmoid(x) | (-0.28, ∞) | YOLOv11 uses this |

**Why ReLU is popular:**
1. Computationally efficient
2. Helps with vanishing gradient
3. Sparse activation (biological plausibility)

**Code Link:** `src/math_from_scratch/activations.py`

---

## 3. Convolutional Neural Networks

### 3.1 Convolution Operation

**Theory:**
Convolution is a mathematical operation that slides a filter (kernel) over an image to detect features like edges, textures, and patterns.

**Mathematics:**

2D Convolution (discrete):
```
(I * K)[i,j] = ΣΣ I[i+m, j+n] × K[m,n]
```

For an input I of size (H, W) and kernel K of size (k_h, k_w):
```
Output size = (H - k_h + 2P)/S + 1, (W - k_w + 2P)/S + 1
```
Where P = padding, S = stride

**Visual Example (3×3 kernel):**
```
Input:           Kernel:         Output:
[1 2 3 4]        [1 0 1]
[5 6 7 8]   *    [0 1 0]    =    [filtered values]
[9 1 2 3]        [1 0 1]
```

**Intuition:**
- Different kernels detect different features
- Edge detection kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
- Blur kernel: [[1,1,1],[1,1,1],[1,1,1]]/9

**Code Link:** `src/math_from_scratch/convolution.py`

---

### 3.2 Pooling Operations

**Theory:**
Pooling reduces spatial dimensions while retaining important information. It provides translation invariance and reduces computation.

**Types:**

**Max Pooling:**
```
output = max(input_region)
```

**Average Pooling:**
```
output = mean(input_region)
```

**Example (2×2 Max Pooling):**
```
Input:           Output:
[1 3 | 2 1]
[5 6 | 3 2]  →   [6 | 3]
─────────        ───────
[2 1 | 0 1]      [2 | 4]
[0 2 | 4 3]
```

**Code Link:** `src/math_from_scratch/pooling.py`

---

## 4. Object Detection Concepts

### 4.1 Intersection over Union (IoU)

**Theory:**
IoU measures how well a predicted bounding box overlaps with the ground truth box.

**Mathematics:**
```
IoU = Area of Intersection / Area of Union

IoU = (A ∩ B) / (A ∪ B)
    = (A ∩ B) / (A + B - A ∩ B)
```

**Visual:**
```
    ┌──────────┐
    │ GT Box   │
    │    ┌─────┼───┐
    │    │INTER│   │
    └────┼─────┘   │
         │ Pred Box│
         └─────────┘
```

**IoU Thresholds:**
- IoU ≥ 0.5: Good detection (PASCAL VOC)
- IoU ≥ 0.75: Strict detection (COCO)
- mAP@50-95: Average over IoU thresholds

**Code Link:** `src/math_from_scratch/metrics.py`

---

### 4.2 Non-Maximum Suppression (NMS)

**Theory:**
NMS eliminates redundant overlapping bounding boxes, keeping only the best prediction for each object.

**Algorithm:**
```
1. Sort all boxes by confidence score
2. Select box with highest confidence
3. Remove all boxes with IoU > threshold with selected box
4. Repeat until no boxes remain
```

**Why needed:**
Large objects may trigger detections in multiple grid cells. NMS keeps only the best one.

**Code Link:** `src/math_from_scratch/nms.py`

---

## 5. YOLO Architecture

### 5.1 YOLO Philosophy

**Theory:**
YOLO (You Only Look Once) treats object detection as a regression problem. Unlike R-CNN which looks at regions, YOLO looks at the entire image once.

**Key Ideas:**
1. Divide image into S×S grid
2. Each grid cell predicts:
   - B bounding boxes (x, y, w, h, confidence)
   - Class probabilities
3. Single forward pass for detection

**Why fast:**
- No region proposals
- Single neural network evaluation
- End-to-end training

---

### 5.2 YOLOv11 Specific Features

**Architecture Components:**

1. **Backbone (CSPDarknet):**
   - Feature extraction
   - Cross-Stage Partial connections
   
2. **Neck (C2PSA + PANet):**
   - Feature aggregation
   - Multi-scale feature fusion
   
3. **Head (Decoupled):**
   - Classification head
   - Regression head
   - Anchor-free design

**YOLOv11 Improvements:**
- C2PSA (Cross-Stage Partial with Spatial Attention)
- Improved efficiency vs accuracy trade-off
- Better small object detection

**Code Link:** `notebooks/04_model_training.ipynb`

---

## 6. Loss Functions

### 6.1 Classification Loss (Cross-Entropy)

**Theory:**
Measures the difference between predicted probability distribution and true distribution.

**Binary Cross-Entropy (BCE):**
```
BCE = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
```

Where y is true label (0 or 1), ŷ is predicted probability.

**Multi-Class Cross-Entropy:**
```
CE = -Σ y_i × log(ŷ_i)
```

**Code Link:** `src/math_from_scratch/loss_functions.py`

---

### 6.2 Box Regression Loss (CIoU)

**Theory:**
CIoU (Complete IoU) considers overlap, center distance, and aspect ratio.

**Mathematics:**

IoU Loss:
```
L_IoU = 1 - IoU
```

DIoU adds center distance:
```
L_DIoU = 1 - IoU + ρ²(b, b_gt) / c²
```
Where ρ is Euclidean distance, c is diagonal of enclosing box.

CIoU adds aspect ratio:
```
L_CIoU = 1 - IoU + ρ²(b, b_gt) / c² + αv
```
Where v measures aspect ratio consistency.

**Code Link:** `src/math_from_scratch/loss_functions.py`

---

## 7. Evaluation Metrics

### 7.1 Precision and Recall

**Mathematics:**
```
Precision = TP / (TP + FP) = "Of all positive predictions, how many are correct?"

Recall = TP / (TP + FN) = "Of all actual positives, how many did we find?"
```

Where:
- TP: True Positives (correct detections)
- FP: False Positives (incorrect detections)
- FN: False Negatives (missed detections)

---

### 7.2 Mean Average Precision (mAP)

**Theory:**
mAP is the primary metric for object detection, averaging precision across recall values and classes.

**Steps:**
1. For each class, compute precision at various recall thresholds
2. Calculate Average Precision (AP) = area under PR curve
3. mAP = mean of AP across all classes

**mAP Variants:**
- mAP@50: Using IoU threshold of 0.5
- mAP@50-95: Average over IoU thresholds 0.5 to 0.95 (step 0.05)

**Code Link:** `src/math_from_scratch/metrics.py`

---

## 8. Training Optimization

### 8.1 Gradient Descent

**Theory:**
Iteratively update parameters in the direction that minimizes loss.

**Mathematics:**
```
θ_new = θ_old - η × ∇L(θ)
```

Where η is learning rate, ∇L is gradient of loss.

**Variants:**
- SGD: Stochastic Gradient Descent
- Adam: Adaptive Moment Estimation
- AdamW: Adam with weight decay

---

### 8.2 Learning Rate Scheduling

**Theory:**
Adjusting learning rate during training for better convergence.

**Common Schedules:**

**Cosine Annealing:**
```
η_t = η_min + 0.5(η_max - η_min)(1 + cos(πt/T))
```

**Linear Warmup:**
```
η_t = η_max × (t / T_warmup)  for t < T_warmup
```

**Code Link:** `notebooks/04_model_training.ipynb`

---

## Updates Log

| Date | Section | Update |
|------|---------|--------|
| 2026-02-06 | Section 1 | Added Task 1 notebook: `01_data_exploration.ipynb` - EDA, class distribution, image statistics |
| 2026-02-06 | Section 1.2 | Added Task 2 notebook: `02_data_preprocessing.ipynb` - Normalization, Augmentation (NumPy) |
| 2026-02-06 | Section 4.1 | Added Task 3 notebook: `03_yolo_format_conversion.ipynb` - Bounding box math, YOLO format |
| 2026-02-06 | Section 1 | Added Task 4 notebook: `04_data_visualization.ipynb` - EDA, class stats, heatmaps |
| 2026-02-06 | Section 2-3 | Added Task 5 notebook: `05_cnn_fundamentals.ipynb` + `src/math_from_scratch/` modules - Convolution, Activations, Pooling |
| 2026-02-06 | Section 4,7 | Added Task 6 notebook: `06_detection_metrics.ipynb` + `metrics.py` - IoU, NMS, Precision/Recall, mAP |
| 2026-02-06 | Section 4 | Added Task 7 notebook: `07_yolo_architecture.ipynb` - Backbone, Neck, Head architecture |
| 2026-02-06 | Section 5 | Added Task 8 notebook: `08_loss_functions.ipynb` + `losses.py` - BCE, Focal, CIoU, YOLO Loss |
| 2026-02-06 | Section 6 | Added Task 9 notebook: `09_yolov11_setup.ipynb` - Ultralytics setup, training config |
| 2026-02-06 | Section 6 | Added Task 10 notebook: `10_model_training.ipynb` - Training, validation, export |
| 2026-02-06 | Section 8 | Added Streamlit app: `streamlit_app/app.py` - Real-time waste classification |

---

*This document will be continuously updated as we progress through each task.*
