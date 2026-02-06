"""
Activation Functions - NumPy Implementation
============================================

Mathematical Foundation:
    Activation functions introduce non-linearity to neural networks,
    enabling them to learn complex patterns.

Code-Theory Link: See docs/CODE-THEORY.md Section 2.2
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    
    Formula: σ(x) = 1 / (1 + e^(-x))
    Range: (0, 1)
    Use: Binary classification, gates in LSTMs
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Derivative: dσ/dx = σ(x) × (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """
    Hyperbolic tangent activation.
    
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Range: (-1, 1)
    """
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative: d(tanh)/dx = 1 - tanh²(x)"""
    return 1 - np.tanh(x)**2


def relu(x):
    """
    Rectified Linear Unit.
    
    Formula: ReLU(x) = max(0, x)
    Range: [0, ∞)
    Use: Most common activation in hidden layers
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative: 1 if x > 0, else 0"""
    return (x > 0).astype(np.float32)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU - prevents "dying ReLU" problem.
    
    Formula: max(αx, x) where α is small (e.g., 0.01)
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Derivative: 1 if x > 0, else α"""
    return np.where(x > 0, 1, alpha)


def silu(x):
    """
    SiLU (Sigmoid Linear Unit) / Swish - USED IN YOLOv11.
    
    Formula: SiLU(x) = x × σ(x)
    Properties: Smooth, non-monotonic, self-gated
    """
    return x * sigmoid(x)


def silu_derivative(x):
    """Derivative: σ(x) + x × σ(x) × (1 - σ(x))"""
    s = sigmoid(x)
    return s + x * s * (1 - s)


def softmax(x, axis=-1):
    """
    Softmax function - converts logits to probabilities.
    
    Formula: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    Output: Probability distribution (sums to 1)
    Use: Multi-class classification output layer
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
