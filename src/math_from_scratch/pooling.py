"""
Pooling Operations - NumPy Implementation
==========================================

Mathematical Foundation:
    Max Pooling: y[i,j] = max_{(m,n) ∈ R} x[m,n]
    Avg Pooling: y[i,j] = (1/|R|) × Σ_{(m,n) ∈ R} x[m,n]
    
    Where R is the pooling region.

Code-Theory Link: See docs/CODE-THEORY.md Section 3.2
"""

import numpy as np


def max_pool2d(image, pool_size=2, stride=2):
    """
    2D Max Pooling operation.
    
    Formula: y[i,j] = max(region)
    Purpose: Downsampling while keeping strongest activations
    
    Args:
        image: Input (H, W) or (H, W, C)
        pool_size: Size of pooling window
        stride: Step size
    
    Returns:
        Pooled output array
    """
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    
    H, W, C = image.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w, C))
    
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = image[i*stride:i*stride+pool_size, 
                              j*stride:j*stride+pool_size, c]
                output[i, j, c] = np.max(region)
    
    return output.squeeze()


def avg_pool2d(image, pool_size=2, stride=2):
    """
    2D Average Pooling operation.
    
    Formula: y[i,j] = mean(region)
    Purpose: Downsampling with smoothing effect
    """
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    
    H, W, C = image.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w, C))
    
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = image[i*stride:i*stride+pool_size,
                              j*stride:j*stride+pool_size, c]
                output[i, j, c] = np.mean(region)
    
    return output.squeeze()


def global_avg_pool2d(image):
    """
    Global Average Pooling - reduces spatial dims to 1×1.
    
    Formula: y[c] = (1/H×W) × Σ_i Σ_j x[i,j,c]
    Use: Before final classification layer
    """
    if len(image.shape) == 2:
        return np.mean(image)
    return np.mean(image, axis=(0, 1))
