# 📚 Code-Theory Linkage — YOLOv8

## Overview

This document connects YOLOv8 code with its theoretical foundations. YOLOv8 is the predecessor to YOLOv11, released in January 2023 by Ultralytics. It introduced the **anchor-free, decoupled-head** paradigm that YOLOv11 later refined.

> **Shared theory**: For fundamentals (CNN, pooling, metrics, loss functions), see [CODE-THEORY.md](CODE-THEORY.md).
> This file covers **YOLOv8-specific** architecture and design choices only.

---

## Table of Contents

- [1. YOLOv8 Architecture Overview](#1-yolov8-architecture-overview)
- [2. Backbone — CSPDarknet with C2f](#2-backbone--cspdarknet-with-c2f)
- [3. Neck — PANet with C2f](#3-neck--panet-with-c2f)
- [4. Head — Decoupled Anchor-Free](#4-head--decoupled-anchor-free)
- [5. Loss Design](#5-loss-design)
- [6. Key Differences from YOLOv11](#6-key-differences-from-yolov11)

---

## 1. YOLOv8 Architecture Overview

```
                        YOLOv8 ARCHITECTURE
┌──────────────────────────────────────────────────────────────┐
│  INPUT (640 × 640 × 3)                                       │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────────────────────────────────┐                   │
│  │  BACKBONE (CSPDarknet + C2f blocks)   │                   │
│  │  Stage 1: Conv 3×3 s=2 → 64ch        │                   │
│  │  Stage 2: C2f → 128ch                │                   │
│  │  Stage 3: C2f → 256ch     (P3)       │                   │
│  │  Stage 4: C2f → 512ch     (P4)       │                   │
│  │  Stage 5: C2f + SPPF → 512ch (P5)    │                   │
│  └───────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────────────────────────────────┐                   │
│  │  NECK (PANet + C2f fusion)            │                   │
│  │  Top-down: P5 → P4 → P3 (upsample)   │                   │
│  │  Bottom-up: P3 → P4 → P5 (downsamp)  │                   │
│  └───────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────────────────────────────────┐                   │
│  │  HEAD (Decoupled, Anchor-Free)        │                   │
│  │  Scale 1: 80×80  (small objects)      │                   │
│  │  Scale 2: 40×40  (medium objects)     │                   │
│  │  Scale 3: 20×20  (large objects)      │                   │
│  │                                       │                   │
│  │  Each scale → cls branch + reg branch │                   │
│  └───────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  NMS → FINAL PREDICTIONS                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Backbone — CSPDarknet with C2f

### 2.1 The C2f Block (Cross-Stage Partial with 2 convolutions, Faster)

**Theory:**
C2f is YOLOv8's signature building block. It improved over YOLOv5's C3 block by creating more gradient flow paths.

**Structure:**
```
Input
  │
  ├── Conv 1×1 (split into two branches)
  │       │
  │       ├── Branch A (direct)
  │       │
  │       └── Branch B → Bottleneck → Bottleneck → ... (n times)
  │                        │             │
  │                        └─ concat ────┘
  │
  └── All branches concatenated → Conv 1×1 → Output
```

**Mathematics:**
```
x_split_a, x_split_b = split(Conv1×1(x))

For i in 1..n:
    x_bottleneck_i = Bottleneck(x_split_b if i==1 else x_bottleneck_{i-1})

Output = Conv1×1(cat(x_split_a, x_split_b, x_bottleneck_1, ... x_bottleneck_n))
```

**Key advantage over C3 (YOLOv5):**
- More gradient paths via concatenation of all intermediate outputs
- Richer feature representation at lower computational cost

**Code Link:** `notebooks/11_yolov8_training.ipynb`

---

### 2.2 SPPF (Spatial Pyramid Pooling Fast)

**Theory:**
SPPF applies sequential max-pooling with small kernel sizes (5×5), which is mathematically equivalent to SPP with large kernels but faster.

```
Input → MaxPool(5) → MaxPool(5) → MaxPool(5)
  │         │            │            │
  └─────────┴────────────┴────────────┘ → Concat → Conv
```

**Why:**
- Captures multi-scale features
- Fixed-size output regardless of input size
- 3× sequential 5×5 ≈ single 13×13 receptive field (faster)

---

## 3. Neck — PANet with C2f

**Theory:**
The neck merges features from different backbone stages. YOLOv8 uses a bidirectional PANet (Path Aggregation Network).

**Flow:**
```
Backbone outputs:
  P3 (80×80)  ←──── small features
  P4 (40×40)  ←──── medium features
  P5 (20×20)  ←──── large features

Top-down path (FPN):
  P5 → Upsample+Concat(P4) → C2f → N4
  N4 → Upsample+Concat(P3) → C2f → N3

Bottom-up path (PAN):
  N3 → Conv s=2 + Concat(N4) → C2f → N4'
  N4'→ Conv s=2 + Concat(P5) → C2f → N5'

Outputs to head: N3, N4', N5'
```

---

## 4. Head — Decoupled Anchor-Free

### 4.1 Decoupled Design

**Theory:**
YOLOv8 separates classification and regression into independent branches (unlike YOLOv5 which couples them).

```
Feature Map (from Neck)
      │
      ├── Classification Branch
      │   Conv → Conv → Sigmoid → class probabilities
      │
      └── Regression Branch
          Conv → Conv → DFL → box coordinates (x, y, w, h)
```

### 4.2 Anchor-Free Detection

**Theory:**
Unlike YOLOv5 which relies on predefined anchor boxes, YOLOv8 directly predicts:
- Center offset (x, y) relative to grid cell
- Width and height via **Distribution Focal Loss (DFL)**

**DFL Mathematics:**
Instead of predicting a single value for each box coordinate, DFL predicts a discrete probability distribution:

```
ŷ = Σ(i=0 to n) P(i) × i

P(i) = softmax(logits)[i]
```

Where `n` is the number of discrete bins (typically 16). This allows the model to express uncertainty about the exact position.

**Advantages over anchors:**
- No anchor hyperparameter tuning
- Faster NMS (fewer candidate boxes)
- Better generalization to unusual aspect ratios

---

## 5. Loss Design

YOLOv8 uses a combination of three losses, same formulations as described in CODE-THEORY.md Section 6:

| Loss Component | Function | Purpose |
|----------------|----------|---------|
| **Classification** | BCE with sigmoid | Class prediction |
| **Box Regression** | CIoU + DFL | Bounding box localisation |
| **Task-Aligned** | TAL (Task-Aligned Learning) | Dynamic label assignment |

### Task-Aligned Assigner (TAL)

**Theory:**
TAL dynamically assigns positive/negative samples during training. Unlike static IoU-based assignment (YOLOv5), TAL considers both classification score and localisation quality:

```
alignment_metric = cls_score^α × IoU^β

where α = 0.5, β = 6.0 (default)
```

High alignment_metric → sample is assigned as positive.

---

## 6. Key Differences from YOLOv11

| Feature | YOLOv8 | YOLOv11 |
|---------|--------|---------|
| Core block | C2f | C2PSA (adds spatial attention) |
| Attention | None | C2PSA spatial attention |
| Label assignment | TAL | TAL (same) |
| Head | Decoupled anchor-free | Decoupled anchor-free |
| Parameters (nano) | ~3.2M | ~2.6M |
| Speed | Baseline | ~5-15% faster |

**Summary:** YOLOv8 and YOLOv11 share the same high-level design. YOLOv11 refines YOLOv8 by adding spatial attention (C2PSA), reducing parameter count, and improving efficiency.

---

## Updates Log

| Date | Section | Update |
|------|---------|--------|
| 2026-02-14 | All | Initial creation — YOLOv8 architecture documentation |
| 2026-02-14 | Notebooks | Added `11_yolov8_training.ipynb` |

---

*See also: [CODE-THEORY.md](CODE-THEORY.md) for shared fundamentals, [CODE-THEORY-V5.md](CODE-THEORY-V5.md) for YOLOv5.*
