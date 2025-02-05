# Solar Panel Crack Detector

A YOLOv11n-based system for detecting cracks in solar panels using computer vision.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RajeevSharma110/solar_panel_detector.git
cd solar_panel_detector
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/MacOS
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

## Project Structure

- `src/`: Source code for the detector
- `config/`: Configuration files
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks with examples

## Usage

1. Prepare your data:
```bash
python -m src.utils.data_preparation --input_dir raw_data --output_dir solar_panel_dataset
```

2. Train the model:
```python
from src.detector import SolarPanelCrackDetector

detector = SolarPanelCrackDetector()
detector.train_model()
```

3. Make predictions:
```python
results = detector.predict('path_to_image.jpg')
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request