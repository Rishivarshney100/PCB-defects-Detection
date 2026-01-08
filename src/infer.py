"""
Inference Script for PCB Defect Detection
Core requirement: Analyzes input image, detects defects, outputs JSON with:
- Defect type classification
- Confidence scores
- Bounding box coordinates
- Defect center (x, y) pixel coordinates
- Severity assessment
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import torch

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.severity_estimator import SeverityEstimator, calculate_defect_center
    from src.utils.visualization import draw_bounding_boxes, save_visualization
except ImportError:
    from severity_estimator import SeverityEstimator, calculate_defect_center
    from utils.visualization import draw_bounding_boxes, save_visualization

CLASS_NAMES = {
    0: "Missing Hole",
    1: "Mouse Bite",
    2: "Open Circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious Copper"
}


class PCBDefectDetector:
    """
    PCB Defect Detection using YOLOv5 model.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, device: str = ""):
        """
        Initialize defect detector.
        
        Args:
            model_path: Path to trained YOLOv5 model (.pt file)
            conf_threshold: Confidence threshold for detections
            device: Device to use ('', 'cpu', 'cuda', '0', etc.)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        self.conf_threshold = conf_threshold
        
        # Determine device
        if device == "":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Initialize severity estimator
        self.severity_estimator = SeverityEstimator()
    
    def predict(self, image_path: str) -> Dict:
        """
        Predict defects in a PCB image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_height, img_width = image.shape[:2]
        
        results = self.model(image, conf=self.conf_threshold)
        defects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = float(x1), float(y1), float(x2), float(y2)
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                defect_type = CLASS_NAMES.get(class_id, f"Class_{class_id}")
                center = calculate_defect_center((xmin, ymin, xmax, ymax))
                severity = self.severity_estimator.estimate_severity(
                    bbox=(xmin, ymin, xmax, ymax),
                    img_width=img_width,
                    img_height=img_height,
                    confidence=confidence
                )
                
                defect_info = {
                    "defect_type": defect_type,
                    "confidence": round(confidence, 4),
                    "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    "center": list(center),
                    "severity": severity
                }
                
                defects.append(defect_info)
        
        output = {
            "image_path": image_path,
            "defects": defects,
            "total_defects": len(defects)
        }
        
        return output
    
    def predict_and_save(self, image_path: str, output_json_path: str = None,
                        output_viz_path: str = None) -> Dict:
        """
        Predict defects and save results.
        
        Args:
            image_path: Path to input image
            output_json_path: Path to save JSON results (optional)
            output_viz_path: Path to save visualization (optional)
            
        Returns:
            Dictionary with detection results
        """
        results = self.predict(image_path)
        
        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_json_path}")
        
        # Save visualization
        if output_viz_path:
            image = cv2.imread(image_path)
            save_visualization(
                image=image,
                defects=results["defects"],
                output_path=output_viz_path
            )
            print(f"Visualization saved to: {output_viz_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="PCB Defect Detection Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="models/weights/best.pt",
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON output (optional)"
    )
    parser.add_argument(
        "--output-viz",
        type=str,
        default=None,
        help="Path to save visualization (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use ('', 'cpu', 'cuda', '0', etc.)"
    )
    
    args = parser.parse_args()
    
    detector = PCBDefectDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    results = detector.predict_and_save(
        image_path=args.image,
        output_json_path=args.output_json,
        output_viz_path=args.output_viz
    )
    
    print("\n" + "="*50)
    print("DEFECT DETECTION RESULTS")
    print("="*50)
    print(f"Image: {results['image_path']}")
    print(f"Total Defects: {results['total_defects']}")
    print("\nDetected Defects:")
    
    if results['total_defects'] == 0:
        print("  No defects detected.")
    else:
        for i, defect in enumerate(results['defects'], 1):
            print(f"\n  Defect {i}:")
            print(f"    Type: {defect['defect_type']}")
            print(f"    Confidence: {defect['confidence']:.4f}")
            print(f"    Center: ({defect['center'][0]}, {defect['center'][1]})")
            print(f"    Bounding Box: {defect['bbox']}")
            print(f"    Severity: {defect['severity']}")
    
    print("\n" + "="*50)
    
    if not args.output_json:
        print("\nJSON Output:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
