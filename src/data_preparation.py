import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import yaml
from tqdm import tqdm


def create_directory_structure(base_path: str = "data") -> None:
    directories = [
        f"{base_path}/images/train",
        f"{base_path}/images/val",
        f"{base_path}/images/test",
        f"{base_path}/labels/train",
        f"{base_path}/labels/val",
        f"{base_path}/labels/test"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        Path(directory).joinpath(".gitkeep").touch()


def split_dataset(images_dir: str,
                 labels_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 seed: int = 42) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in os.listdir(images_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    random.shuffle(image_files)
    
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    
    base_path = Path(images_dir).parent
    
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for img_file in tqdm(files, desc=f"Moving {split_name} files"):
            src_img = Path(images_dir) / img_file
            dst_img = base_path / "images" / split_name / img_file
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            label_file = Path(img_file).stem + ".txt"
            src_label = Path(labels_dir) / label_file
            dst_label = base_path / "labels" / split_name / label_file
            
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                dst_label.touch()


def convert_voc_to_yolo(voc_xml_path: str, yolo_txt_path: str,
                        class_mapping: dict, img_width: int, img_height: int) -> None:
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree.ElementTree required for VOC conversion")
    
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue
        
        class_id = class_mapping[class_name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        center_x = ((xmin + xmax) / 2.0) / img_width
        center_y = ((ymin + ymax) / 2.0) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    os.makedirs(os.path.dirname(yolo_txt_path), exist_ok=True)
    with open(yolo_txt_path, 'w') as f:
        f.writelines(yolo_annotations)


def validate_dataset(data_dir: str) -> Tuple[bool, List[str]]:
    errors = []
    data_path = Path(data_dir)
    
    required_dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_path in required_dirs:
        if not (data_path / dir_path).exists():
            errors.append(f"Missing directory: {dir_path}")
    
    for split in ["train", "val", "test"]:
        images_dir = data_path / "images" / split
        labels_dir = data_path / "labels" / split
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        image_files = {f.stem for f in images_dir.glob("*.*") 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}}
        label_files = {f.stem for f in labels_dir.glob("*.txt")}
        
        missing_labels = image_files - label_files
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} images without labels")
        
        orphan_labels = label_files - image_files
        if orphan_labels:
            errors.append(f"{split}: {len(orphan_labels)} labels without images")
    
    return len(errors) == 0, errors


def create_dataset_yaml(output_path: str = "data/dataset.yaml",
                       class_names: List[str] = None) -> None:
    if class_names is None:
        class_names = ["missing_component", "misalignment", "solder_defect", "no_defect"]
    
    config = {
        'path': './data',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    print("Creating directory structure...")
    create_directory_structure()
    print("\nCreating dataset.yaml...")
    create_dataset_yaml()
    print("\nDataset preparation setup complete!")
