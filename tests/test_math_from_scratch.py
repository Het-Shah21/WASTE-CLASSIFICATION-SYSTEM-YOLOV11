"""
Test Suite for Math From Scratch Module
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from math_from_scratch import activations, convolution, pooling, metrics, losses


class TestActivations:
    """Test activation functions."""
    
    def test_relu(self):
        x = np.array([-1, 0, 1, 2])
        result = activations.relu(x)
        expected = np.array([0, 0, 1, 2])
        assert np.allclose(result, expected), f"ReLU failed: {result}"
        print("✅ ReLU test passed")
    
    def test_sigmoid(self):
        x = np.array([0])
        result = activations.sigmoid(x)
        assert np.isclose(result[0], 0.5), f"Sigmoid(0) should be 0.5, got {result}"
        print("✅ Sigmoid test passed")
    
    def test_silu(self):
        x = np.array([0])
        result = activations.silu(x)
        assert np.isclose(result[0], 0.0), f"SiLU(0) should be 0, got {result}"
        print("✅ SiLU test passed")


class TestMetrics:
    """Test detection metrics."""
    
    def test_iou_perfect(self):
        box = [0, 0, 100, 100]
        iou = metrics.calculate_iou(box, box)
        assert np.isclose(iou, 1.0), f"IoU of same box should be 1, got {iou}"
        print("✅ IoU (perfect) test passed")
    
    def test_iou_no_overlap(self):
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        iou = metrics.calculate_iou(box1, box2)
        assert np.isclose(iou, 0.0), f"IoU of non-overlapping should be 0, got {iou}"
        print("✅ IoU (no overlap) test passed")


class TestLosses:
    """Test loss functions."""
    
    def test_bce(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0.9, 0.1])
        loss = losses.binary_cross_entropy(y_true, y_pred)
        assert loss > 0, f"BCE should be positive, got {loss}"
        print("✅ BCE test passed")
    
    def test_focal(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0.9, 0.1])
        loss = losses.focal_loss(y_true, y_pred)
        assert loss > 0, f"Focal loss should be positive, got {loss}"
        print("✅ Focal loss test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("🧪 Running Math From Scratch Tests")
    print("="*50 + "\n")
    
    test_acts = TestActivations()
    test_acts.test_relu()
    test_acts.test_sigmoid()
    test_acts.test_silu()
    
    test_metrics = TestMetrics()
    test_metrics.test_iou_perfect()
    test_metrics.test_iou_no_overlap()
    
    test_losses = TestLosses()
    test_losses.test_bce()
    test_losses.test_focal()
    
    print("\n" + "="*50)
    print("✅ All tests passed!")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()
