"""
Severity Estimator Module
Calculates defect severity based on bounding box area and confidence score.
"""

import numpy as np
from typing import Tuple, Dict


class SeverityEstimator:
    """
    Estimates defect severity based on:
    - Defect area relative to image size
    - Confidence score
    """
    
    def __init__(self, 
                 low_threshold: float = 0.01,
                 medium_threshold: float = 0.05,
                 confidence_weight: float = 0.3):
        """
        Initialize severity estimator.
        
        Args:
            low_threshold: Area ratio threshold for low severity (default: 1%)
            medium_threshold: Area ratio threshold for medium severity (default: 5%)
            confidence_weight: Weight for confidence score in severity calculation
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.confidence_weight = confidence_weight
    
    def calculate_area_ratio(self, bbox: Tuple[float, float, float, float], 
                            img_width: int, img_height: int) -> float:
        """
        Calculate defect area relative to image size.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in pixel coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Area ratio (0-1)
        """
        xmin, ymin, xmax, ymax = bbox
        defect_area = (xmax - xmin) * (ymax - ymin)
        image_area = img_width * img_height
        return defect_area / image_area if image_area > 0 else 0.0
    
    def estimate_severity(self, 
                         bbox: Tuple[float, float, float, float],
                         img_width: int,
                         img_height: int,
                         confidence: float = 1.0) -> str:
        """
        Estimate defect severity.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in pixel coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            confidence: Detection confidence score (0-1)
            
        Returns:
            Severity level: "Low", "Medium", or "High"
        """
        area_ratio = self.calculate_area_ratio(bbox, img_width, img_height)
        
        adjusted_low = self.low_threshold * (1 - self.confidence_weight * (1 - confidence))
        adjusted_medium = self.medium_threshold * (1 - self.confidence_weight * (1 - confidence))
        
        if area_ratio < adjusted_low:
            return "Low"
        elif area_ratio < adjusted_medium:
            return "Medium"
        else:
            return "High"
    
    def get_severity_score(self, 
                          bbox: Tuple[float, float, float, float],
                          img_width: int,
                          img_height: int,
                          confidence: float = 1.0) -> float:
        """
        Get numerical severity score (0-1) for ranking.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax) in pixel coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            confidence: Detection confidence score (0-1)
            
        Returns:
            Severity score (0-1), where 1 is most severe
        """
        area_ratio = self.calculate_area_ratio(bbox, img_width, img_height)
        severity_score = area_ratio * 0.7 + confidence * 0.3
        return min(1.0, severity_score * 10)


def calculate_defect_center(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """
    Calculate the center coordinates of a defect bounding box.
    
    Args:
        bbox: Bounding box (xmin, ymin, xmax, ymax) in pixel coordinates
        
    Returns:
        Center coordinates (x, y) as integers
    """
    xmin, ymin, xmax, ymax = bbox
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)
    return (center_x, center_y)
