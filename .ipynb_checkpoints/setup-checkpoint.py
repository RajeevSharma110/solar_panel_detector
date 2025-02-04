from setuptools import setup, find_packages

setup(
    name="solar_panel_detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ultralytics>=8.0.0",
        "pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "pyyaml>=5.4.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "python-dotenv>=0.19.0",
            "jupyter>=1.0.0",
        ]
    },
    author="RajeevSharma110",
    author_email="rajeevsharma_ds@hotmail.com",
    description="A YOLOv11-based solar panel crack detection system",
    keywords="computer-vision, solar-panels, crack-detection, yolo",
    python_requires=">=3.8",
)