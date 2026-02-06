"""
Convolution Operations - NumPy Implementation
==============================================

Mathematical Foundation:
    2D Convolution: (I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] × K[m,n]
    
    Output Size: O = (I - K + 2P) / S + 1
    Where: I=input size, K=kernel size, P=padding, S=stride

Code-Theory Link: See docs/CODE-THEORY.md Section 3.1
"""

import numpy as np


def conv2d(image, kernel, stride=1, padding=0):
    """
    2D Convolution operation (NumPy only).
    
    Mathematical operation:
        (I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] × K[m,n]
    
    Args:
        image: Input image (H, W) or (H, W, C)
        kernel: Convolution kernel (k_h, k_w)
        stride: Step size for sliding window
        padding: Zero-padding around image
    
    Returns:
        Convolved output array
    """
    # Handle grayscale vs RGB
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    
    H, W, C = image.shape
    k_h, k_w = kernel.shape
    
    # Apply padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        H, W, C = image.shape
    
    # Calculate output dimensions
    out_h = (H - k_h) // stride + 1
    out_w = (W - k_w) // stride + 1
    
    # Initialize output
    output = np.zeros((out_h, out_w, C))
    
    # Perform convolution
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                region = image[i*stride:i*stride+k_h, j*stride:j*stride+k_w, c]
                output[i, j, c] = np.sum(region * kernel)
    
    return output.squeeze()


# Common Kernels
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

LAPLACIAN = np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]], dtype=np.float32)

GAUSSIAN_3x3 = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], dtype=np.float32) / 16

SHARPEN = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]], dtype=np.float32)
