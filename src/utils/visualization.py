"""
Visualization utilities for defect detection results.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import os


# Color palette for different defect classes
COLORS = {
    'missing_hole': (0, 0, 255),           # Red
    'mouse_bite': (255, 165, 0),           # Orange
    'open_circuit': (255, 0, 255),         # Magenta
    'short': (0, 255, 255),                # Cyan
    'spur': (255, 192, 203),               # Pink
    'spurious_copper': (0, 255, 0),        # Green
    'default': (255, 255, 0)               # Yellow
}

# Class name mapping
CLASS_NAMES = {
    0: 'Missing Hole',
    1: 'Mouse Bite',
    2: 'Open Circuit',
    3: 'Short',
    4: 'Spur',
    5: 'Spurious Copper'
}


def draw_bounding_boxes(image: np.ndarray,
                       defects: List[Dict],
                       class_names: Dict[int, str] = None,
                       show_confidence: bool = True,
                       show_center: bool = True,
                       show_severity: bool = True) -> np.ndarray:
    """
    Draw bounding boxes, labels, and defect centers on image.
    
    Args:
        image: Input image (numpy array, BGR format)
        defects: List of defect dictionaries with keys:
                 - 'bbox': [xmin, ymin, xmax, ymax]
                 - 'defect_type': str or class_id
                 - 'confidence': float
                 - 'center': [x, y] (optional)
                 - 'severity': str (optional)
        class_names: Dictionary mapping class_id to class name
        show_confidence: Whether to show confidence scores
        show_center: Whether to mark defect centers
        show_severity: Whether to show severity levels
        
    Returns:
        Annotated image (numpy array)
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    annotated_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    for defect in defects:
        bbox = defect['bbox']
        xmin, ymin, xmax, ymax = [int(coord) for coord in bbox]
        
        defect_type = defect.get('defect_type', 'unknown')
        if isinstance(defect_type, int):
            defect_type_name = class_names.get(defect_type, f'Class {defect_type}')
            class_key = list(class_names.keys())[list(class_names.values()).index(defect_type_name)] if defect_type_name in class_names.values() else 'default'
        else:
            defect_type_name = defect_type
            class_key = defect_type.lower().replace(' ', '_')
            if class_key not in COLORS:
                class_key = 'default'
        
        color = COLORS.get(class_key, COLORS['default'])
        
        thickness = 2
        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), color, thickness)
        
        label_parts = [defect_type_name]
        
        if show_confidence and 'confidence' in defect:
            confidence = defect['confidence']
            label_parts.append(f'{confidence:.2f}')
        
        if show_severity and 'severity' in defect:
            severity = defect['severity']
            label_parts.append(f'[{severity}]')
        
        label = ' | '.join(label_parts)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        
        label_y = max(ymin, text_height + 10)
        cv2.rectangle(
            annotated_image,
            (xmin, label_y - text_height - 10),
            (xmin + text_width + 10, label_y + baseline),
            color,
            -1
        )
        
        cv2.putText(
            annotated_image,
            label,
            (xmin + 5, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )
        
        if show_center:
            if 'center' in defect:
                center_x, center_y = defect['center']
            else:
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)
            
            cv2.circle(annotated_image, (center_x, center_y), 5, color, -1)
            cv2.circle(annotated_image, (center_x, center_y), 8, (255, 255, 255), 2)
    
    return annotated_image


def save_visualization(image: np.ndarray,
                      defects: List[Dict],
                      output_path: str,
                      **kwargs) -> None:
    """
    Save annotated image to file.
    
    Args:
        image: Input image (numpy array)
        defects: List of defect dictionaries
        output_path: Path to save the annotated image
        **kwargs: Additional arguments for draw_bounding_boxes
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    annotated_image = draw_bounding_boxes(image, defects, **kwargs)
    cv2.imwrite(output_path, annotated_image)


def create_summary_visualization(images: List[np.ndarray],
                                defects_list: List[List[Dict]],
                                output_path: str,
                                grid_size: Tuple[int, int] = None) -> None:
    """
    Create a grid visualization of multiple images with defects.
    
    Args:
        images: List of input images
        defects_list: List of defect lists (one per image)
        output_path: Path to save the grid visualization
        grid_size: (rows, cols) for grid layout. Auto-calculated if None.
    """
    if not images:
        return
    
    num_images = len(images)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    # Resize images to same size for grid
    target_size = (320, 320)
    resized_images = []
    
    for img, defects in zip(images, defects_list):
        annotated = draw_bounding_boxes(img, defects)
        resized = cv2.resize(annotated, target_size)
        resized_images.append(resized)
    
    # Create grid
    grid_height = rows * target_size[1]
    grid_width = cols * target_size[0]
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        y_start = row * target_size[1]
        x_start = col * target_size[0]
        grid_image[y_start:y_start + target_size[1], 
                  x_start:x_start + target_size[0]] = img
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid_image)
