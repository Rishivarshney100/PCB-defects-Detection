"""
Metrics calculation utilities for defect detection evaluation.
"""

import numpy as np
from typing import List, Tuple, Dict


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Bounding box (xmin, ymin, xmax, ymax)
        box2: Bounding box (xmin, ymin, xmax, ymax)
        
    Returns:
        IoU score (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_precision_recall(predictions: List[Dict],
                              ground_truth: List[Dict],
                              iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Calculate precision and recall for defect detection.
    
    Args:
        predictions: List of predicted defects with 'bbox' key
        ground_truth: List of ground truth defects with 'bbox' key
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (precision, recall)
    """
    if len(predictions) == 0 and len(ground_truth) == 0:
        return 1.0, 1.0
    
    if len(predictions) == 0:
        return 0.0, 0.0
    
    if len(ground_truth) == 0:
        return 0.0, 1.0
    
    # Match predictions to ground truth
    matched_gt = set()
    true_positives = 0
    
    for pred in predictions:
        pred_bbox = tuple(pred['bbox'])
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            gt_bbox = tuple(gt['bbox'])
            iou = calculate_iou(pred_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    precision = true_positives / len(predictions) if len(predictions) > 0 else 0.0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    return precision, recall


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_ap(precision_recall_curve: List[Tuple[float, float]]) -> float:
    """
    Calculate Average Precision (AP) from precision-recall curve.
    
    Args:
        precision_recall_curve: List of (precision, recall) tuples
        
    Returns:
        Average Precision score
    """
    if not precision_recall_curve:
        return 0.0
    
    # Sort by recall
    sorted_curve = sorted(precision_recall_curve, key=lambda x: x[1])
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for r in np.arange(0, 1.1, 0.1):
        # Find maximum precision at recall >= r
        max_precision = 0.0
        for prec, rec in sorted_curve:
            if rec >= r:
                max_precision = max(max_precision, prec)
        ap += max_precision
    
    return ap / 11.0
