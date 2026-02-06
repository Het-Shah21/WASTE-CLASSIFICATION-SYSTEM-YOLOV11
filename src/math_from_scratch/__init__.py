"""
Math From Scratch - NumPy Implementations
==========================================

Educational implementations of core ML/DL concepts using only NumPy.

Modules:
    - convolution: 2D convolution operations
    - activations: Activation functions (ReLU, SiLU, etc.)
    - pooling: Pooling operations (Max, Avg, Global)
    - metrics: Detection metrics (IoU, NMS, mAP)
    - losses: Loss functions (BCE, Focal, CIoU)

Usage:
    from src.math_from_scratch import convolution, activations
    from src.math_from_scratch.metrics import calculate_iou
"""

from . import convolution
from . import activations
from . import pooling
from . import metrics
from . import losses

__all__ = [
    'convolution',
    'activations', 
    'pooling',
    'metrics',
    'losses',
]

__version__ = '1.0.0'
