import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from .utils.data_preparation import prepare_dataset

class SolarPanelCrackDetector:
    """
    A class for training and detecting cracks in solar panels using YOLO.
    """

    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize the detector with configuration.
        :param config_path: Path to the configuration YAML file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.prepare_data()

    def prepare_data(self):
        """
        Prepare the dataset for training.
        """
        prepare_dataset(
            self.config['images_path'],
            self.config['annotations_path'],
            self.config['dataset_path']
        )

    def train_model(self):
        """
        Train the YOLOv11 model using the dataset.
        """
        model = YOLO("yolo11n.pt")  # Ensure correct YOLO version
        
        model.train(
            data=os.path.join(self.config['dataset_path'], 'dataset.yaml'),
            epochs=self.config['epochs'],
            imgsz=640,
            batch=16,
            project="runs/train",   # Explicitly define where to save
            name="solar_crack_train"  # Ensure it doesn’t overwrite
        )
        model.export(format="torchscript")  # Save as TorchScript model
        model_path = os.path.join(self.config['model_save_path'], "best.pt")
        print(f"Model saved at {model_path}")

    def predict(self, image_path):
        """
        Perform prediction on a new image.
        :param image_path: Path to the image to be analyzed.
        :return: Detection results (bounding boxes).
        """
        #model_path = f"/home/antonio/Documents/JupyterNotebooks/Rajeev_Sharma/solar_panel_detector/runs/train/{self.config['run_name']}/weights/best.pt"
        model_path = "/home/antonio/Documents/JupyterNotebooks/Rajeev_Sharma/solar_panel_detector/runs/train/solar_crack_train/weights/best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        #model = YOLO("/home/antonio/Documents/JupyterNotebooks/Rajeev_Sharma/solar_panel_detector/runs/detect/solar_crack_detection/weights/best.pt")
        model = YOLO(model_path)
        results = model.predict(image_path)
        return results[0].boxes.data # Extract bounding box information

if __name__ == "__main__":
    detector = SolarPanelCrackDetector("config/config.yaml")
    detector.train_model()  # Train the model
    results = detector.predict("/home/antonio/Documents/JupyterNotebooks/Rajeev_Sharma/solar_panel_detector/Electroluminescence+(EL)+Testing+for+PV+Modules.jpg")  # Test prediction
    print(results)
