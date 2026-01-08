"""
Evaluation Script for PCB Defect Detection
Calculates mAP, precision, recall, F1-score, and performance metrics.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package required. Install with: pip install ultralytics")


# Class name mapping (must match dataset.yaml)
CLASS_NAMES = {
    0: "Missing Hole",
    1: "Mouse Bite",
    2: "Open Circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious Copper"
}


class ModelEvaluator:
    """
    Evaluates YOLOv5 model performance on test dataset.
    """
    
    def __init__(self, model_path: str, data_yaml: str, device: str = ""):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model (.pt file)
            data_yaml: Path to dataset.yaml
            device: Device to use
        """
        if device == "":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.data_yaml = data_yaml
    
    def evaluate(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict:
        """
        Evaluate model on test set using YOLOv5 built-in validation.
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating model on test set...")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        
        results = self.model.val(
            data=self.data_yaml,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            plots=True
        )
        
        metrics = {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            "f1_score": 0.0
        }
        
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / \
                                 (metrics["precision"] + metrics["recall"])
        
        if hasattr(results.box, 'maps'):
            metrics["per_class_AP50"] = {
                CLASS_NAMES.get(i, f"Class_{i}"): float(ap)
                for i, ap in enumerate(results.box.maps) if i < len(CLASS_NAMES)
            }
        
        return metrics
    
    def measure_inference_speed(self, image_paths: List[str], num_runs: int = 100) -> Dict:
        """
        Measure inference speed (FPS and average time).
        
        Args:
            image_paths: List of image paths for testing
            num_runs: Number of inference runs
            
        Returns:
            Dictionary with performance metrics
        """
        if not image_paths:
            test_dir = Path("data/images/test")
            if test_dir.exists():
                image_paths = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                image_paths = [str(p) for p in image_paths[:min(10, len(image_paths))]]
        
        if not image_paths:
            return {"fps": 0.0, "avg_time_ms": 0.0, "device": self.device}
        
        print(f"\nMeasuring inference speed on {len(image_paths)} images ({num_runs} runs)...")
        
        test_image = cv2.imread(image_paths[0])
        for _ in range(5):
            _ = self.model(test_image, verbose=False)
        
        times = []
        for _ in tqdm(range(num_runs), desc="Running inference"):
            img_path = np.random.choice(image_paths)
            image = cv2.imread(img_path)
            
            start_time = time.time()
            _ = self.model(image, verbose=False)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)
        
        avg_time_ms = np.mean(times)
        fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0
        
        return {
            "fps": round(fps, 2),
            "avg_time_ms": round(avg_time_ms, 2),
            "min_time_ms": round(np.min(times), 2),
            "max_time_ms": round(np.max(times), 2),
            "std_time_ms": round(np.std(times), 2),
            "device": self.device
        }
    
    def generate_confusion_matrix(self, test_images_dir: str, 
                                  output_path: str = "results/confusion_matrix.png") -> None:
        """
        Generate confusion matrix from test predictions.
        
        Args:
            test_images_dir: Directory containing test images
            output_path: Path to save confusion matrix plot
        """
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
        print(f"Test directory not found: {test_images_dir}")
            return
    
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if not image_files:
            print("No test images found")
            return
        
        print(f"\nGenerating confusion matrix from {len(image_files)} test images...")
        print("Note: Full confusion matrix requires ground truth annotations.")
        print("Run model.val() with plots=True to generate confusion matrix.")
    
    def save_evaluation_report(self, metrics: Dict, output_path: str = "results/evaluation_report.json") -> None:
        """
        Save evaluation report to JSON file.
        
        Args:
            metrics: Dictionary with evaluation metrics
            output_path: Path to save report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nEvaluation report saved to: {output_path}")


def print_evaluation_summary(metrics: Dict):
    """
    Print formatted evaluation summary.
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nDetection Metrics:")
    print(f"  mAP@0.5:     {metrics.get('mAP50', 0.0):.4f}")
    print(f"  mAP@0.5:0.95: {metrics.get('mAP50-95', 0.0):.4f}")
    print(f"  Precision:   {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:      {metrics.get('recall', 0.0):.4f}")
    print(f"  F1-Score:    {metrics.get('f1_score', 0.0):.4f}")
    
    if 'per_class_AP50' in metrics:
        print("\nPer-Class AP@0.5:")
        for class_name, ap in metrics['per_class_AP50'].items():
            print(f"  {class_name}: {ap:.4f}")
    
    if 'performance' in metrics:
        perf = metrics['performance']
        print("\nPerformance Metrics:")
        print(f"  FPS:            {perf.get('fps', 0.0):.2f}")
        print(f"  Avg Time:       {perf.get('avg_time_ms', 0.0):.2f} ms")
        print(f"  Device:         {perf.get('device', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PCB Defect Detection Model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/weights/best.pt",
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.yaml",
        help="Path to dataset.yaml"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_report.json",
        help="Path to save evaluation report"
    )
    parser.add_argument(
        "--speed-test",
        action="store_true",
        help="Run inference speed test"
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        model_path=args.model,
        data_yaml=args.data,
        device=args.device
    )
    
    metrics = evaluator.evaluate(
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    if args.speed_test:
        perf_metrics = evaluator.measure_inference_speed([])
        metrics["performance"] = perf_metrics
    
    print_evaluation_summary(metrics)
    evaluator.save_evaluation_report(metrics, args.output)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
