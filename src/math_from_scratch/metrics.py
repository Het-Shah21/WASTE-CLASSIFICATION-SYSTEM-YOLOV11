"""
Object Detection Metrics - NumPy Implementation
================================================

Mathematical Foundations:
    IoU = Intersection / Union
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    mAP = Mean of Average Precision across classes

Code-Theory Link: See docs/CODE-THEORY.md Section 7
"""

import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.
    
    Formula: IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        box1, box2: [x1, y1, x2, y2] format (corners)
    
    Returns:
        IoU value in [0, 1]
    """
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    
    # Intersection
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    
    # Union
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - intersection
    
    return intersection / (union + 1e-6)


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression.
    
    Algorithm:
        1. Sort boxes by score (descending)
        2. Select highest scoring box
        3. Remove boxes with IoU > threshold
        4. Repeat until done
    
    Args:
        boxes: Array of [x1, y1, x2, y2], shape (N, 4)
        scores: Confidence scores, shape (N,)
        iou_threshold: Suppression threshold
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(sorted_indices) > 0:
        best_idx = sorted_indices[0]
        keep.append(best_idx)
        sorted_indices = sorted_indices[1:]
        
        if len(sorted_indices) == 0:
            break
        
        remaining_boxes = boxes[sorted_indices]
        ious = np.array([calculate_iou(boxes[best_idx], box) for box in remaining_boxes])
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[mask]
    
    return keep


def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calculate precision and recall.
    
    Returns:
        precision, recall, f1_score
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0, 0, 0
    
    gt_matched = [False] * len(gt_boxes)
    tp, fp = 0, 0
    
    for pred_box in pred_boxes:
        best_iou, best_gt_idx = 0, -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = sum(1 for m in gt_matched if not m)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def calculate_ap(precisions, recalls):
    """
    Calculate Average Precision (11-point interpolation).
    
    Formula: AP = (1/11) × Σ max(p(r)) for r ∈ [0, 0.1, ..., 1.0]
    """
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    sorted_idx = np.argsort(recalls)
    recalls = recalls[sorted_idx]
    precisions = precisions[sorted_idx]
    
    ap = 0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precisions[mask])
    
    return ap / 11
