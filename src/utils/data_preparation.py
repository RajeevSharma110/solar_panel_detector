import os
import json
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np

def load_and_convert_annotation(json_path, image_shape):
    """
    Load JSON annotation and convert to YOLO format
    
    Args:
        json_path: Path to JSON annotation file
        image_shape: Tuple of (height, width) of the image
    
    Returns:
        list: YOLO format annotations (one per cell)
    """
    height, width = image_shape
    yolo_annotations = []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Process each instance (cell) in the JSON
    for instance in data["instances"]:
        corners = instance["corners"]
        
        # Calculate bounding box from corners
        x_coords = [point["x"] for point in corners]
        y_coords = [point["y"] for point in corners]
        
        # Calculate bounding box coordinates
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convert to YOLO format (normalized coordinates)
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        center_x = (x_min + box_width * width / 2) / width
        center_y = (y_min + box_height * height / 2) / height
        
        # Class ID: 0 for normal cell, 1 for defected cell
        class_id = 1 if instance.get("defected_module", False) else 0
        
        # YOLO format: <class> <x_center> <y_center> <width> <height>
        yolo_annotation = f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        yolo_annotations.append(yolo_annotation)
    
    return yolo_annotations

def prepare_dataset(images_dir, annotations_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Prepare dataset by converting annotations and organizing files
    
    Args:
        images_dir: Directory containing images
        annotations_dir: Directory containing JSON annotations
        output_dir: Output directory for YOLO format dataset
        split_ratio: Tuple of (train, val, test) ratios
    """
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Create dataset.yaml
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'normal_cell',
            1: 'defected_cell'
        }
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f)

    # Get list of images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    np.random.shuffle(image_files)

    # Calculate split indices
    n_images = len(image_files)
    n_train = int(n_images * split_ratio[0])
    n_val = int(n_images * split_ratio[1])
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    # Process each split
    for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for img_file in files:
            # Get corresponding JSON file
            json_file = os.path.splitext(img_file)[0] + '.json'
            json_path = os.path.join(annotations_dir, json_file)
            
            if not os.path.exists(json_path):
                print(f"Warning: No annotation found for {img_file}")
                continue
            
            # Read image to get dimensions
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
            
            # Convert annotations
            yolo_annotations = load_and_convert_annotation(json_path, img.shape[:2])
            
            # Save image
            shutil.copy(img_path, os.path.join(output_dir, 'images', split, img_file))
            
            # Save annotations
            label_path = os.path.join(output_dir, 'labels', split, 
                                    os.path.splitext(img_file)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

if __name__ == "__main__":
    prepare_dataset(
        images_dir='dataset_1/images',
        annotations_dir='dataset_1/annotations',
        output_dir='solar_panel_dataset'
    )