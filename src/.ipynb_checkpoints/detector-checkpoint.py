import os
from pathlib import Path
import yaml
from ultralytics import YOLO
from .utils.data_preparation import prepare_dataset

class SolarPanelCrackDetector:
    def __init__(self, config_path='config/config.yaml'):
        Initialize the detector with configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.prepare_data()
        
    def prepare_data(self):
        Prepare the dataset
        prepare_dataset(
            self.config['images_path'],
            self.config['annotations_path'],
            self.config['dataset_path']
        )

    def train_model(self):
        Train the YOLOv11n model
        model = YOLO('yolov11n.pt')
        
        model.train(
            data=os.path.join(self.config['dataset_path'], 'dataset.yaml'),
            epochs=self.config['epochs'],
            imgsz=640,
            batch=16,
            name=self.config['run_name']
        )
        
        model.save(self.config['model_save_path'])

    def predict(self, image_path):
        Perform prediction on a new image
        model = YOLO(self.config['model_save_path'])
        results = model.predict(image_path)
        return results[0].boxes.data
