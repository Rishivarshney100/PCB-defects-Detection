"""
Training Script for PCB Defect Detection using YOLOv5
"""

import os
import shutil
import argparse
from pathlib import Path
import torch


def train_yolov5(data_yaml: str = "data/dataset.yaml",
                 epochs: int = 100,
                 batch_size: int = 16,
                 img_size: int = 640,
                 model_size: str = "s",
                 device: str = "",
                 project: str = "runs/train",
                 name: str = "pcb_defect_detection",
                 patience: int = 50,
                 save_period: int = 10) -> None:
    """
    Train YOLOv5 model for PCB defect detection.
    
    Args:
        data_yaml: Path to dataset.yaml configuration file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training (square)
        model_size: YOLOv5 model size ('s', 'm', 'l', 'x')
        device: Device to use ('', 'cpu', '0', '0,1', etc.)
        project: Project directory for saving results
        name: Experiment name
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    
    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Training YOLOv5{model_size} on {data_yaml}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    
    model_name = f"yolov5{model_size}.pt"
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    best_model_path = Path(project) / name / "weights" / "best.pt"
    if best_model_path.exists():
        os.makedirs("models/weights", exist_ok=True)
        shutil.copy2(best_model_path, "models/weights/best.pt")
        print(f"\nBest model saved to: models/weights/best.pt")
    
    print("\nTraining completed!")
    print(f"Results saved to: {Path(project) / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv5 for PCB Defect Detection")
    parser.add_argument("--data", type=str, default="data/dataset.yaml",
                       help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Image size")
    parser.add_argument("--model", type=str, default="s",
                       choices=["n", "s", "m", "l", "x"],
                       help="YOLOv5 model size")
    parser.add_argument("--device", type=str, default="",
                       help="Device (cuda, cpu, or '' for auto)")
    parser.add_argument("--project", type=str, default="runs/train",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="pcb_defect_detection",
                       help="Experiment name")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    train_yolov5(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_size=args.model,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience
    )
