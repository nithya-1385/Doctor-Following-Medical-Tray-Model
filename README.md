# ML_Project
Autonomous Computer Vision Based Human-Following Robot [Problem statement 19]

# Doctor Detection and Human-Following Simulation with YOLOv8

This project demonstrates a complete pipeline for detecting doctors in image sequences and simulating human-following behavior using YOLOv8. It includes dataset preparation, annotation conversion, training, and visual logging of movement commands.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Movement Logic](#movement-logic)
- [Training Configuration](#training-configuration)
- [Customization](#customization)
- [Key Functions](#key-functions)
- [Video Output](#video-output)
- [Logging](#logging)
- [Hardware Requirements](#hardware-requirements)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [Dataset Used](#dataset-used)
- [License](#license)
- [Contact](#contact)  

---

## Overview

This project implements an end-to-end workflow that:
- Converts CVAT XML annotations to YOLO format
- Trains a YOLOv8 nano model for person detection
- Generates robot movement commands (FORWARD/STOP) based on detected position changes
- Creates annotated tracking videos with visual feedback
- Logs all movement decisions for analysis

## Features

- **Automated Dataset Preparation**: Converts XML annotations to YOLO format and creates train/val/test splits
- **GPU-Accelerated Training**: Leverages CUDA for fast YOLOv8 model training
- **Intelligent Tracking Logic**: Detects vertical movement and generates appropriate robot commands
- **Visual Output**: Produces annotated videos showing detection boxes and movement commands
- **Comprehensive Logging**: Records all movement decisions with timestamps

## Requirements

```bash
pip install ultralytics opencv-python-headless lxml torch
```

## Usage

### 1. Setup and Installation

```python
# Install required packages
!pip install -q ultralytics opencv-python-headless lxml

# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
```

### 2. Upload Dataset

Upload your `archive.zip` containing the dataset when prompted by the notebook.

### 3. Run the Pipeline

The notebook automatically executes:
- XML to YOLO conversion
- Dataset splitting (70% train, 15% val, 15% test)
- Model training (30 epochs, batch size 8)
- Detection and command generation
- Video creation with annotations

### 4. Download Results

Results are automatically zipped and downloaded as `output5.zip`.

## Output Structure

```
output/
├── splits/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   └── test/
├── labels_yolo/          # Converted YOLO format labels
├── weights/              # Trained model weights
├── tracking_1.mp4        # Annotated video (main test set)
├── tracking_2.mp4        # Annotated video (additional test folder)
├── movement_commands.txt # Complete command log
└── data.yaml             # YOLO training configuration
```

## Movement Logic

The system uses a simple yet effective tracking algorithm:

- **FORWARD**: Doctor moved significantly downward (>3 pixels in y-axis)
- **STOP**: No significant movement detected for 5 consecutive frames
- **Missing Detection Handling**: Continues FORWARD for up to 3 frames if detection is lost

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8 nano (`yolov8n.pt`) |
| Epochs | 30 |
| Image Size | 640x640 |
| Batch Size | 8 |
| Confidence Threshold | 0.25 |

## Customization

### Adjust Movement Sensitivity

```python
# In detect_and_command_topdown function
forward_threshold = 3  # Increase for less sensitive movement detection
stop_frame_limit = 5   # Number of frames before issuing STOP command
max_miss_allowed = 3   # Frames to continue FORWARD if detection is lost
```

### Modify Training Parameters

```python
model.train(
    data=YOLO_DATA_YAML,
    epochs=30,      # Increase for better accuracy
    imgsz=640,      # Image size
    batch=8         # Adjust based on GPU memory
)
```

## Key Functions

| Function | Description |
|----------|-------------|
| `frame_to_filename()` | Converts frame IDs to standardized filenames |
| `parse_xml_to_yolo()` | Converts CVAT XML annotations to YOLO format |
| `copy_for_split()` | Organizes dataset into train/val/test splits |
| `detect_and_command_topdown()` | Core detection and command logic |

## Video Output

Generated videos include:
- Red bounding boxes around detected persons
- White text overlay showing current command (FORWARD/STOP)
- 10 FPS playback for smooth visualization

## Logging

The `movement_commands.txt` file contains:
- Timestamp of execution
- Dataset path
- Frame-by-frame decisions for all splits
- Video output locations

## Hardware Requirements

- **GPU**: Recommended (Tesla T4 or better)
- **RAM**: 12GB+ recommended
- **Storage**: ~2GB for outputs

## Notes

- Designed for Google Colab environment
- Automatically detects and uses GPU if available
- Handles missing detections gracefully
- Supports multiple test folders for validation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No images found | Verify your dataset path matches `DATASET_DIR` variable |
| Out of memory | Reduce batch size in training configuration |
| Poor detection | Increase training epochs or adjust confidence threshold |

## License

This project uses YOLOv8 from Ultralytics (AGPL-3.0 license).

## Dataset Used 

### Details

- **Source**: [Medical Staff People Tracking](https://www.kaggle.com/datasets/trainingdatapro/medical-staff-people-tracking)
- **URL**: https://www.kaggle.com/datasets/trainingdatapro/medical-staff-people-tracking
- **License**: Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
- **Description**: The dataset contains a collection of frames extracted from videos captured within a hospital environment. The bounding boxes are drawn around the doctors, nurses, and other people who appear in the video footage.
- **Size**: 274 images, 2 xml files, 2 csv files

### Acknowledgments

We thank Unique Data(Owner) [Kaggle] for providing the dataset used in this project.

## Contact

For questions or issues, please open an issue on GitHub.

## Authors:
Nithya Prashaanthi. R and Neeraja Kumar
