"""
Loss Functions - NumPy Implementation
=====================================

YOLO Loss Components:
    L_total = λ_box × L_box + λ_cls × L_cls + λ_obj × L_obj

Code-Theory Link: See docs/CODE-THEORY.md Section 5
"""

import numpy as np


def binary_cross_entropy(y_true, y_pred, epsilon=1e-7):
    """
    Binary Cross-Entropy Loss.
    
    Formula: BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-7):
    """
    Focal Loss for class imbalance.
    
    Formula: FL = -α(1-p_t)^γ × log(p_t)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weight = (1 - p_t) ** gamma
    loss = -alpha_t * focal_weight * np.log(p_t)
    return np.mean(loss)


def ciou_loss(box_pred, box_gt):
    """
    Complete IoU Loss.
    
    Formula: CIoU = IoU - (ρ²/c²) - αv
    
    Args:
        box_pred, box_gt: [x1, y1, x2, y2] format
    """
    # IoU
    inter_x1 = max(box_pred[0], box_gt[0])
    inter_y1 = max(box_pred[1], box_gt[1])
    inter_x2 = min(box_pred[2], box_gt[2])
    inter_y2 = min(box_pred[3], box_gt[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union_area = area_pred + area_gt - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    # Center distance
    c_pred = [(box_pred[0] + box_pred[2])/2, (box_pred[1] + box_pred[3])/2]
    c_gt = [(box_gt[0] + box_gt[2])/2, (box_gt[1] + box_gt[3])/2]
    rho2 = (c_pred[0] - c_gt[0])**2 + (c_pred[1] - c_gt[1])**2
    
    # Enclosing box diagonal
    enc_x1 = min(box_pred[0], box_gt[0])
    enc_y1 = min(box_pred[1], box_gt[1])
    enc_x2 = max(box_pred[2], box_gt[2])
    enc_y2 = max(box_pred[3], box_gt[3])
    c2 = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-6
    
    # Aspect ratio term
    w_pred = box_pred[2] - box_pred[0]
    h_pred = box_pred[3] - box_pred[1]
    w_gt = box_gt[2] - box_gt[0]
    h_gt = box_gt[3] - box_gt[1]
    v = (4 / np.pi**2) * (np.arctan(w_gt/(h_gt+1e-6)) - np.arctan(w_pred/(h_pred+1e-6)))**2
    alpha = v / (1 - iou + v + 1e-6)
    
    ciou = iou - (rho2 / c2) - alpha * v
    return 1 - ciou


class YOLOLoss:
    """Complete YOLO Loss combining box, class, and objectness losses."""
    
    def __init__(self, lambda_box=0.05, lambda_cls=0.5, lambda_obj=1.0):
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
    
    def __call__(self, pred_boxes, gt_boxes, pred_cls, gt_cls, pred_obj, gt_obj):
        box_loss = np.mean([ciou_loss(p, g) for p, g in zip(pred_boxes, gt_boxes)])
        cls_loss = binary_cross_entropy(gt_cls, pred_cls)
        obj_loss = binary_cross_entropy(gt_obj, pred_obj)
        
        total = self.lambda_box * box_loss + self.lambda_cls * cls_loss + self.lambda_obj * obj_loss
        return {'total': total, 'box': box_loss, 'cls': cls_loss, 'obj': obj_loss}
