# 📚 Code-Theory Linkage — YOLOv5

## Overview

This document covers the **YOLOv5-specific** architecture and theory. YOLOv5 was released in June 2020 by Ultralytics and remains the most production-deployed YOLO variant. It is the **anchor-based** predecessor to the anchor-free YOLOv8/v11.

> **Shared theory**: For CNN, pooling, metrics, and loss function fundamentals, see [CODE-THEORY.md](CODE-THEORY.md).

---

## Table of Contents

- [1. YOLOv5 Architecture Overview](#1-yolov5-architecture-overview)
- [2. Backbone — CSPDarknet53 + Focus](#2-backbone--cspdarknet53--focus)
- [3. Neck — SPP + PANet](#3-neck--spp--panet)
- [4. Head — Coupled Anchor-Based](#4-head--coupled-anchor-based)
- [5. Loss Design](#5-loss-design)
- [6. Key Differences from YOLOv8 and YOLOv11](#6-key-differences-from-yolov8-and-yolov11)

---

## 1. YOLOv5 Architecture Overview

```
                        YOLOv5 ARCHITECTURE
┌──────────────────────────────────────────────────────────────┐
│  INPUT (640 × 640 × 3)                                       │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────────────────────────────────┐                   │
│  │  BACKBONE (CSPDarknet53 + Focus)      │                   │
│  │  Stage 0: Focus (slice) → 64ch       │                   │
│  │  Stage 1: CSP + C3 → 128ch           │                   │
│  │  Stage 2: CSP + C3 → 256ch    (P3)   │                   │
│  │  Stage 3: CSP + C3 → 512ch    (P4)   │                   │
│  │  Stage 4: CSP + C3 + SPP → 1024ch(P5)│                   │
│  └───────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────────────────────────────────┐                   │
│  │  NECK (PANet with C3 blocks)          │                   │
│  │  Top-down: P5 → P4 → P3 (upsample)   │                   │
│  │  Bottom-up: P3 → P4 → P5 (downsamp)  │                   │
│  └───────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────────────────────────────────┐                   │
│  │  HEAD (Coupled, ANCHOR-BASED)         │                   │
│  │  Scale 1: 80×80 × 3 anchors          │                   │
│  │  Scale 2: 40×40 × 3 anchors          │                   │
│  │  Scale 3: 20×20 × 3 anchors          │                   │
│  │                                       │                   │
│  │  Output per anchor: [x,y,w,h,obj,cls] │                   │
│  └───────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  NMS → FINAL PREDICTIONS                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Backbone — CSPDarknet53 + Focus

### 2.1 Focus Layer

**Theory:**
YOLOv5's unique contribution. Instead of an initial strided convolution, the Focus layer **slices** the image into 4 sub-images using a stride-2 pattern, then concatenates and convolves.

**Mathematics:**
```
Input: (H, W, 3)

Slice[0] = Input[0::2, 0::2, :]  → (H/2, W/2, 3)
Slice[1] = Input[1::2, 0::2, :]  → (H/2, W/2, 3)
Slice[2] = Input[0::2, 1::2, :]  → (H/2, W/2, 3)
Slice[3] = Input[1::2, 1::2, :]  → (H/2, W/2, 3)

Concat → (H/2, W/2, 12)
Conv 3×3 → (H/2, W/2, 64)
```

**Why:**
- No information loss (every pixel is preserved)
- Equivalent to a strided convolution but more efficient
- Better for small object features

> **Note:** Later YOLOv5 versions (v6.0+) replaced Focus with a 6×6 strided convolution for deployment simplicity. The Ultralytics `yolov5nu.pt` uses this approach.

---

### 2.2 C3 Block (CSP Bottleneck with 3 Convolutions)

**Theory:**
The C3 block is YOLOv5's core building block. It uses Cross-Stage Partial design with 3 convolution layers.

**Structure:**
```
Input
  │
  ├── Conv 1×1 → Bottleneck × n → (Branch A)
  │
  └── Conv 1×1 ──────────────────── (Branch B)
  │
  Concat(A, B) → Conv 1×1 → Output
```

**Comparison with C2f (YOLOv8):**

| Feature | C3 (YOLOv5) | C2f (YOLOv8) |
|---------|-------------|--------------|
| Gradient paths | 2 branches | All intermediate outputs |
| Concatenation | 2 tensors | n+2 tensors |
| Feature richness | Moderate | Higher |
| Speed | Slightly faster | Slightly slower |

---

### 2.3 SPP (Spatial Pyramid Pooling)

**Theory:**
Applies max-pooling with multiple kernel sizes (5, 9, 13) and concatenates the results. This captures multi-scale spatial information.

```
Input → MaxPool(5×5)  ─┐
      → MaxPool(9×9)  ─┼── Concat → Conv
      → MaxPool(13×13)─┘
      → Identity ──────┘
```

**Difference from SPPF (YOLOv8/v11):**
SPP uses parallel pooling with different kernel sizes. SPPF uses sequential pooling with the same small kernel (5×5) three times — mathematically equivalent but faster.

---

## 3. Neck — PANet with C3

**Theory:**
Same PANet (Path Aggregation Network) concept as v8/v11, but using C3 blocks for feature fusion.

```
P5 (20×20) → Upsample → Concat(P4) → C3 → N4
N4          → Upsample → Concat(P3) → C3 → N3
N3          → Conv s=2 → Concat(N4) → C3 → N4'
N4'         → Conv s=2 → Concat(P5) → C3 → N5'
```

---

## 4. Head — Coupled Anchor-Based

### 4.1 Anchor-Based Detection

**Theory:**
YOLOv5 uses **predefined anchor boxes** at each scale. Each grid cell uses 3 anchors of different aspect ratios.

**Default Anchors (COCO):**
```
Scale 1 (80×80): [10,13], [16,30], [33,23]        # small objects
Scale 2 (40×40): [30,61], [62,45], [59,119]       # medium objects
Scale 3 (20×20): [116,90], [156,198], [373,326]   # large objects
```

**Prediction per anchor:**
```
Output = [tx, ty, tw, th, objectness, class_0, class_1]
         ├──── 4 ────┤  ├── 1 ───┤  ├──── nc ────┤

Total per anchor = 4 + 1 + num_classes
```

**Box decoding:**
```
bx = σ(tx) × 2 - 0.5 + cx         (center x)
by = σ(ty) × 2 - 0.5 + cy         (center y)
bw = (σ(tw) × 2)² × pw            (width)
bh = (σ(th) × 2)² × ph            (height)
```

Where:
- `cx, cy` = grid cell offset
- `pw, ph` = anchor width/height
- `σ` = sigmoid function

### 4.2 Coupled Head

Unlike YOLOv8/v11's decoupled head, YOLOv5 uses a single convolutional layer per scale that outputs all predictions (class + box + objectness) together.

```
Feature Map
     │
     Conv 1×1 → (B, anchors × (5 + nc), H, W)
                  ├─ tx, ty, tw, th (box)
                  ├─ objectness
                  └─ class scores
```

---

## 5. Loss Design

### 5.1 Three Loss Components

```
L_total = λ_box × L_box + λ_obj × L_obj + λ_cls × L_cls
```

| Loss | Function | Details |
|------|----------|---------|
| **L_box** | CIoU | Bounding box regression |
| **L_obj** | BCE | Objectness score (does anchor contain object?) |
| **L_cls** | BCE | Class prediction |

### 5.2 Label Assignment

**IoU-based static assignment:**
```
1. For each ground-truth box, compute IoU with all anchors
2. Assign GT to anchor with highest IoU
3. Also assign to anchors with IoU > threshold
4. Remaining anchors are negative samples
```

**Contrast with TAL (YOLOv8/v11):**

| Feature | IoU-based (YOLOv5) | TAL (YOLOv8/v11) |
|---------|-------------------|-------------------|
| Static vs Dynamic | Static | Dynamic |
| Considers cls score | No | Yes |
| Adaptation | Fixed rules | Learns assignment |

### 5.3 Objectness Loss

YOLOv5 has an **explicit objectness branch** that predicts whether each anchor contains an object. YOLOv8/v11 removed this — they rely on implicit objectness via the classification score.

```
L_obj = BCE(predicted_objectness, IoU_with_gt)
```

---

## 6. Key Differences from YOLOv8 and YOLOv11

| Feature | YOLOv5 | YOLOv8 | YOLOv11 |
|---------|--------|--------|---------|
| **Release** | 2020 | 2023 | 2024 |
| **Detection** | Anchor-based | Anchor-free | Anchor-free |
| **Head** | Coupled | Decoupled | Decoupled |
| **Core block** | C3 | C2f | C2PSA |
| **Stem** | Focus layer | Conv 3×3 s=2 | Conv 3×3 s=2 |
| **Pooling** | SPP (parallel) | SPPF (sequential) | SPPF |
| **Objectness** | Explicit | Implicit | Implicit |
| **Assignment** | IoU-static | TAL-dynamic | TAL-dynamic |
| **Box loss** | CIoU | CIoU + DFL | CIoU + DFL |
| **Parameters (n)** | ~2.5M | ~3.2M | ~2.6M |

**Summary:**
- **YOLOv5**: Battle-tested, anchor-based, simpler design, smallest nano variant
- **YOLOv8**: Modern anchor-free, DFL for better localisation, richer features via C2f
- **YOLOv11**: Adds spatial attention (C2PSA), fewer params with same/better accuracy

---

## Updates Log

| Date | Section | Update |
|------|---------|--------|
| 2026-02-14 | All | Initial creation — YOLOv5 architecture documentation |
| 2026-02-14 | Notebooks | Added `12_yolov5_training.ipynb` |

---

*See also: [CODE-THEORY.md](CODE-THEORY.md) for shared fundamentals, [CODE-THEORY-V8.md](CODE-THEORY-V8.md) for YOLOv8.*
