# 📊 YOLO Version Comparison — v5 vs v8 vs v11

## Overview

A comprehensive, side-by-side comparison of YOLOv5, YOLOv8, and YOLOv11 across all major technical dimensions. All three versions are from **Ultralytics** and share a common lineage, but differ significantly in architecture, performance, and design philosophy.

---

## Table of Contents

- [1. Architecture Comparison](#1-architecture-comparison)
- [2. Performance Benchmarks](#2-performance-benchmarks)
- [3. Design Philosophy](#3-design-philosophy)
- [4. Training Characteristics](#4-training-characteristics)
- [5. Deployment & Production](#5-deployment--production)
- [6. Feature-by-Feature Matrix](#6-feature-by-feature-matrix)
- [7. When to Use Which](#7-when-to-use-which)
- [8. Final Recommendation](#8-final-recommendation)

---

## 1. Architecture Comparison

### 1.1 High-Level Structure

| Component | YOLOv5 (2020) | YOLOv8 (2023) | YOLOv11 (2024) |
|-----------|:-------------:|:-------------:|:--------------:|
| **Backbone** | CSPDarknet53 | CSPDarknet (modified) | CSPDarknet (modified) |
| **Core Block** | C3 | C2f | C2PSA |
| **Stem** | Focus layer | Conv 3×3 stride-2 | Conv 3×3 stride-2 |
| **Spatial Pooling** | SPP (parallel) | SPPF (sequential) | SPPF (sequential) |
| **Neck** | PANet + C3 | PANet + C2f | PANet + C2f/C2PSA |
| **Head Type** | Coupled | Decoupled | Decoupled |
| **Detection** | Anchor-based (3/scale) | Anchor-free | Anchor-free |
| **Attention** | None | None | C2PSA (spatial) |

### 1.2 Core Block Evolution

```
YOLOv5 (C3)              YOLOv8 (C2f)             YOLOv11 (C2PSA)
┌───────────┐            ┌───────────┐            ┌───────────────┐
│  Input    │            │  Input    │            │   Input       │
│    │      │            │    │      │            │     │         │
│  ┌─┴─┐   │            │  ┌─┴─┐   │            │   ┌─┴─┐      │
│  │   │   │            │  │   │   │            │   │   │      │
│ Conv Conv │            │ Conv Conv │            │  Conv Conv   │
│  │   │   │            │  │   │ │ │            │   │   │ │    │
│  │  Btn×n│            │  │  Btn Btn│            │   │ Btn+Attn│
│  │   │   │            │  │   │ │ │            │   │   │ │    │
│  Concat  │            │  Concat(all)            │   Concat(all)│
│    │     │            │    │      │            │     │        │
│  Conv    │            │  Conv     │            │   Conv       │
└───────────┘            └───────────┘            └───────────────┘
  2 paths                 n+2 paths               n+2 paths + attn
```

### 1.3 Detection Head

```
YOLOv5 (Coupled + Anchors)        YOLOv8/v11 (Decoupled + Anchor-Free)
┌────────────────────┐             ┌──────────────────────────┐
│  Feature Map       │             │  Feature Map             │
│       │            │             │       │                  │
│  Conv 1×1          │             │   ┌───┴───┐              │
│       │            │             │   │       │              │
│ [tx,ty,tw,th,      │             │ CLS Branch  REG Branch   │
│  obj, c0, c1]      │             │   │          │           │
│                    │             │ [c0, c1]   [x,y,w,h]    │
│ Per anchor × 3     │             │                          │
└────────────────────┘             └──────────────────────────┘
```

---

## 2. Performance Benchmarks

### 2.1 Model Size & Speed (Nano Variants)

| Metric | YOLOv5n | YOLOv8n | YOLOv11n |
|--------|:-------:|:-------:|:--------:|
| **Parameters** | ~2.5M | ~3.2M | ~2.6M |
| **FLOPs** | ~7.1G | ~8.7G | ~6.5G |
| **Model Size (MB)** | ~5.3 | ~6.3 | ~5.4 |
| **Inference (ms, GPU)** | ~6.3 | ~6.2 | ~5.3 |
| **Inference (ms, CPU)** | ~73 | ~80 | ~52 |

### 2.2 Accuracy (COCO val2017, Nano)

| Metric | YOLOv5n | YOLOv8n | YOLOv11n |
|--------|:-------:|:-------:|:--------:|
| **mAP@50** | 45.7% | 52.6% | 54.7% |
| **mAP@50-95** | 28.0% | 37.3% | 39.5% |

### 2.3 Accuracy vs Speed Trade-off

```
mAP@50-95
    ▲
40% │                            ★ v11n
    │                        ★ v8n
38% │
    │
36% │
    │
    │
30% │
28% │    ★ v5n
    │
    └────────────────────────────────────▶ Inference Speed
    fast                              slow
```

### 2.4 Model Variant Comparison (All Scales)

| Scale | YOLOv5 mAP | YOLOv8 mAP | YOLOv11 mAP |
|-------|:----------:|:----------:|:-----------:|
| **Nano (n)** | 28.0 | 37.3 | 39.5 |
| **Small (s)** | 37.4 | 44.9 | 47.0 |
| **Medium (m)** | 45.4 | 50.2 | 51.5 |
| **Large (l)** | 49.0 | 52.9 | 53.4 |
| **XLarge (x)** | 50.7 | 53.9 | 54.7 |

---

## 3. Design Philosophy

### 3.1 Anchor-Based vs Anchor-Free

| Aspect | Anchor-Based (v5) | Anchor-Free (v8, v11) |
|--------|:------------------:|:---------------------:|
| **Setup** | Requires anchor tuning | No hyperparameters |
| **NMS speed** | Slower (more candidates) | Faster |
| **Generalization** | Limited by anchor priors | Better for unusual shapes |
| **Recall** | Can miss unusual shapes | Better coverage |
| **Complexity** | Higher | Lower |

### 3.2 Label Assignment

| Aspect | Static IoU (v5) | TAL Dynamic (v8, v11) |
|--------|:--------------:|:---------------------:|
| Considers cls score | ❌ | ✅ |
| Adapts during training | ❌ | ✅ |
| Handles ambiguity | Poorly | Well |
| Implementation | Simpler | More complex |

### 3.3 Objectness Branch

| Aspect | YOLOv5 | YOLOv8/v11 |
|--------|:------:|:----------:|
| **Has objectness** | ✅ Explicit | ❌ Removed |
| **Output per anchor** | 5 + nc | 4 + nc |
| **Rationale** | Learn what is/isn't an object | Cls score is sufficient |

---

## 4. Training Characteristics

### 4.1 Loss Functions

| Component | YOLOv5 | YOLOv8 | YOLOv11 |
|-----------|:------:|:------:|:-------:|
| **Box Loss** | CIoU | CIoU + DFL | CIoU + DFL |
| **Cls Loss** | BCE (sigmoid) | BCE (sigmoid) | BCE (sigmoid) |
| **Obj Loss** | BCE (sigmoid) | ❌ (removed) | ❌ (removed) |
| **Total** | 3 components | 2 components | 2 components |

### 4.2 Augmentation Pipeline

All three versions share the same Ultralytics augmentation pipeline:

| Augmentation | Supported |
|-------------|:---------:|
| Mosaic (4-image) | ✅ |
| MixUp | ✅ |
| HSV jitter | ✅ |
| Random flip | ✅ |
| Scale/translate | ✅ |
| Copy-paste | ✅ |

### 4.3 Training Speed

| Aspect | YOLOv5n | YOLOv8n | YOLOv11n |
|--------|:-------:|:-------:|:--------:|
| **Epochs to converge** | ~250-300 | ~200-250 | ~150-200 |
| **Time per epoch (typical)** | Baseline | ~1.05× | ~0.95× |
| **Memory usage** | Lowest | Moderate | Moderate |

---

## 5. Deployment & Production

### 5.1 Export Formats

All three support identical export targets via Ultralytics:

| Format | v5 | v8 | v11 |
|--------|:--:|:--:|:---:|
| ONNX | ✅ | ✅ | ✅ |
| TensorRT | ✅ | ✅ | ✅ |
| CoreML | ✅ | ✅ | ✅ |
| TFLite | ✅ | ✅ | ✅ |
| OpenVINO | ✅ | ✅ | ✅ |
| Edge TPU | ✅ | ✅ | ✅ |

### 5.2 Production Readiness

| Aspect | YOLOv5 | YOLOv8 | YOLOv11 |
|--------|:------:|:------:|:-------:|
| **Maturity** | 🟢 Highest | 🟡 High | 🟠 Moderate |
| **Community size** | 🟢 Largest | 🟡 Large | 🟠 Growing |
| **Production deployments** | 🟢 Thousands | 🟡 Hundreds | 🟠 Early |
| **Bug reports resolved** | 🟢 Extensive | 🟡 Good | 🟠 Ongoing |
| **Documentation** | 🟢 Comprehensive | 🟡 Good | 🟠 Good |
| **Third-party tools** | 🟢 Most | 🟡 Many | 🟠 Some |

### 5.3 Edge Device Compatibility

| Device | YOLOv5n | YOLOv8n | YOLOv11n |
|--------|:-------:|:-------:|:--------:|
| **Raspberry Pi 4** | ✅ (~200ms) | ✅ (~220ms) | ✅ (~180ms) |
| **Jetson Nano** | ✅ (~25ms) | ✅ (~28ms) | ✅ (~22ms) |
| **Mobile (TFLite)** | ✅ (~40ms) | ✅ (~45ms) | ✅ (~35ms) |

---

## 6. Feature-by-Feature Matrix

| Feature | YOLOv5 | YOLOv8 | YOLOv11 |
|---------|:------:|:------:|:-------:|
| Anchor-free | ❌ | ✅ | ✅ |
| Decoupled head | ❌ | ✅ | ✅ |
| Spatial attention | ❌ | ❌ | ✅ |
| DFL (Distribution Focal Loss) | ❌ | ✅ | ✅ |
| Task-Aligned Assignment | ❌ | ✅ | ✅ |
| Focus layer | ✅ | ❌ | ❌ |
| Explicit objectness | ✅ | ❌ | ❌ |
| Multi-task (det+seg+cls+pose) | ✅ | ✅ | ✅ |
| Python API | ✅ | ✅ | ✅ |
| CLI | ✅ | ✅ | ✅ |
| Tracking integration | ❌ | ✅ | ✅ |
| Active maintenance | ✅ | ✅ | ✅ |

---

## 7. When to Use Which

### 🟢 Use YOLOv5 When:

- **Production stability** is the top priority
- Deploying to **constrained edge devices** with limited toolchain support
- You need the **most battle-tested** option
- Working with legacy systems that already integrate YOLOv5
- **Minimal model size** matters more than accuracy

### 🔵 Use YOLOv8 When:

- You want a **balanced** option (good accuracy + good maturity)
- Migrating from YOLOv5 and want **modern architecture** benefits
- You need **anchor-free** detection without bleeding-edge risk
- **Community support** and third-party tool compatibility matter
- You want DFL for better **bounding-box localisation**

### 🟣 Use YOLOv11 When:

- **Maximum accuracy** is the priority
- You want the **best speed-accuracy trade-off**
- Working on tasks where **spatial attention** helps (e.g., cluttered scenes)
- **Small object detection** is important
- You are starting a **new project** and want the latest improvements

---

## 8. Final Recommendation

### For This Waste Classification Project:

> **🏆 Recommended: YOLOv11**

| Criterion | Winner | Why |
|-----------|:------:|-----|
| **Accuracy** | YOLOv11 | Highest mAP across all scales |
| **Speed** | YOLOv11 | Fastest inference (CPU and GPU) |
| **Efficiency** | YOLOv11 | Fewer params than v8, close to v5 |
| **Small objects** | YOLOv11 | C2PSA attention helps with small waste items |
| **Production readiness** | YOLOv5 | Most mature, but v11 is stable enough |

### Decision Matrix (Scored 1-5):

| Criterion (Weight) | YOLOv5 | YOLOv8 | YOLOv11 |
|--------------------|:------:|:------:|:-------:|
| Accuracy (30%) | 3 | 4 | **5** |
| Speed (20%) | 4 | 3 | **5** |
| Model size (10%) | **5** | 3 | 4 |
| Production maturity (15%) | **5** | 4 | 3 |
| Feature set (15%) | 3 | 4 | **5** |
| Community (10%) | **5** | 4 | 3 |
| **Weighted Total** | **3.85** | **3.70** | **4.35** |

### Summary

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  For BEST ACCURACY + SPEED  →  YOLOv11  ★★★★★         │
│  For BALANCED + SAFE choice →  YOLOv8   ★★★★☆         │
│  For MAX STABILITY          →  YOLOv5   ★★★★☆         │
│                                                        │
│  All three are excellent for waste classification.     │
│  The differences on a 2-class problem are small.       │
│  Choose based on your deployment constraints.          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

> **Note:** On a simple 2-class task like waste classification (Organic vs Recyclable), all three versions will achieve very similar real-world accuracy. The architectural differences matter more on complex datasets (COCO, 80+ classes). For this project, any version will work well — but YOLOv11 gives you the best starting point.

---

*Generated: 2026-02-14 | See: [CODE-THEORY.md](CODE-THEORY.md), [CODE-THEORY-V8.md](CODE-THEORY-V8.md), [CODE-THEORY-V5.md](CODE-THEORY-V5.md)*
