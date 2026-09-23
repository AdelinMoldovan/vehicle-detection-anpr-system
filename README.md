# Vehicle Detection, ANPR & Speed Estimation System

A computer vision system for detecting and tracking vehicles, estimating their speed, and automatically recognizing license plates from video footage.

The project combines **YOLOv8**, **ByteTrack**, **EasyOCR**, perspective transformation, and post-processing techniques into a unified vehicle analysis pipeline.

## Overview

The system processes an input video frame by frame and detects vehicles using YOLOv8. Each detected vehicle is assigned a persistent tracking ID using ByteTrack.

From this point, processing is divided into two main branches:

- **Speed Estimation** – estimates the speed of each tracked vehicle using its position and perspective-corrected movement.
- **Automatic Number Plate Recognition (ANPR)** – detects license plates, associates them with tracked vehicles, and extracts the registration number using OCR.

The results from both branches are then combined using the vehicle's tracking ID.

Final results are exported as:

- an annotated output video;
- a CSV file containing the processed vehicle data.

---

## Processing Pipeline

```text
                         Input Video
                              |
                        Frame Extraction
                              |
                    YOLOv8 Vehicle Detection
                              |
                          ByteTrack
                              |
                     Unique Vehicle ID
                              |
              +---------------+---------------+
              |                               |
       SPEED ESTIMATION                  ANPR PIPELINE
              |                               |
       Vehicle Position              YOLOv8 Plate Detection
              |                               |
    Perspective Transformation        Plate-Vehicle Association
              |                               |
       Position History                  Preprocessing
              |                               |
    Displacement Calculation                EasyOCR
              |                               |
       Speed Estimation               Reading Filtering
              |                               |
              |                      Voting-based Validation
              |                               |
              +---------------+---------------+
                              |
                  Integration by Tracker ID
                              |
                       Post-processing
                              |
                        Interpolation
                              |
                  +-----------+-----------+
                  |                       |
          final_results.csv        final_video.mp4
```

---

## Features

- Vehicle detection using **YOLOv8**
- Multi-object vehicle tracking using **ByteTrack**
- Persistent unique ID assignment for detected vehicles
- Automatic license plate detection
- Association of license plates with their corresponding vehicles
- License plate recognition using **EasyOCR**
- OCR result filtering
- Voting-based license plate validation across multiple frames
- Vehicle speed estimation
- Perspective transformation for improved spatial measurements
- Vehicle position history and displacement calculation
- Integration of ANPR and speed data using tracking IDs
- Post-processing and interpolation of results
- Annotated video generation
- CSV result export

---

## Technologies

The project is implemented in **Python** and uses computer vision and machine learning technologies including:

- Python
- OpenCV
- YOLOv8
- ByteTrack
- EasyOCR
- NumPy
- Pandas

Additional dependencies can be found in `requirements.txt`.

---

## Project Structure

```text
vehicle-detection-anpr-system/
│
├── anpr/
│   ├── detect.py
│   ├── interpolate.py
│   ├── ocr.py
│   ├── plate_utils.py
│   └── visualize.py
│
├── combined/
│   ├── anpr_manager.py
│   ├── combined_detector.py
│   ├── combined_visualizer.py
│   └── result_manager.py
│
├── models/
│   └── vehicle_detector.py
│
├── speed/
│   └── speed_estimator.py
│
├── main.py
├── speed_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

Large model weights, datasets, input videos, and generated outputs are intentionally excluded from the repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AdelinMoldovan/vehicle-detection-anpr-system.git
```

Navigate to the project directory:

```bash
cd vehicle-detection-anpr-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How It Works

### 1. Vehicle Detection

Each video frame is processed using a YOLOv8 vehicle detection model to locate vehicles in the scene.

### 2. Vehicle Tracking

ByteTrack tracks detected vehicles between consecutive frames and assigns each vehicle a persistent unique ID.

This tracking ID serves as the link between the speed estimation and ANPR pipelines.

### 3. Speed Estimation

For each tracked vehicle, its position is recorded over time.

A perspective transformation is applied to map image coordinates into a more suitable spatial representation. The system maintains a history of vehicle positions, calculates displacement, and uses this information to estimate vehicle speed.

### 4. License Plate Recognition

License plates are detected using YOLOv8 and associated with their corresponding tracked vehicles.

Detected plate regions are preprocessed before being passed to EasyOCR.

Because OCR readings may vary between frames, the system filters the readings and uses voting-based validation to determine a more reliable license plate result.

### 5. Data Integration

Speed and ANPR information are associated using the vehicle's ByteTrack ID.

This allows information collected independently by the two processing branches to be combined into a single result for each vehicle.

### 6. Post-processing

The collected results are post-processed and interpolated where necessary before generating the final outputs.

---

## Output

The system produces two primary outputs:

### `final_results.csv`

Contains the processed information collected for tracked vehicles.

### `final_video.mp4`

An annotated version of the processed video containing the visualization of detection and analysis results.

---

## Architecture

The project follows a modular architecture separating:

- vehicle detection;
- vehicle tracking;
- speed estimation;
- ANPR processing;
- OCR validation;
- result management;
- visualization.

This makes individual components easier to test, modify, and extend independently.

---

## Future Improvements

Potential extensions include:

- real-time video stream processing;
- improved speed calibration;
- support for multiple camera configurations;
- improved license plate recognition under difficult lighting conditions;
- database integration;
- automatic event logging;
- web-based visualization and monitoring.

---

## Author

**Adelin Moldovan**

Developed as part of a dissertation project focused on computer vision-based vehicle analysis.
