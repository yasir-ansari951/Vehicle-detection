# Vehicle Detection in Adverse Weather Conditions

## Overview

Vehicle Detection in Adverse Weather Conditions is a computer vision project designed to detect and identify vehicles in challenging environmental conditions such as rain, fog, snow, haze, and low-light situations. The system leverages deep learning and object detection techniques to improve road safety, traffic monitoring, and intelligent transportation systems.

## Features

* Real-time vehicle detection
* Detection under adverse weather conditions
* Support for multiple vehicle types
* Deep learning-based object detection
* Image and video processing
* Performance evaluation using standard metrics
* Easy-to-use interface

## Technologies Used

* Python
* OpenCV
* TensorFlow / PyTorch
* NumPy
* Pandas
* Matplotlib
* YOLO (You Only Look Once) Object Detection

## Dataset

The project uses a vehicle dataset containing images captured under various weather conditions, including:

* Rainy weather
* Foggy weather
* Snowy weather
* Hazy conditions
* Low-light and nighttime environments

Dataset files are not included in this repository due to GitHub file size limitations.

## Project Structure

```text
Vehicle Detection System/
│
├── dataset/
│   └── DAWN/
│
├── models/
│
├── results/
│
├── src/
│
├── README.md
└── requirements.txt
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/yasir-ansari951/Vehicle-detection.git
cd Vehicle-detection
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Run the main application:

```bash
python main.py
```

or

```bash
python app.py
```

(Use the appropriate entry file based on your project structure.)

## Methodology

1. Data Collection
2. Data Preprocessing
3. Weather Condition Enhancement
4. Vehicle Detection using Deep Learning Models
5. Performance Evaluation
6. Result Visualization

## Performance Metrics

The model is evaluated using:

* Precision
* Recall
* F1-Score
* Mean Average Precision (mAP)
* Detection Accuracy

## Applications

* Intelligent Transportation Systems
* Smart Cities
* Traffic Monitoring
* Autonomous Vehicles
* Road Safety Analysis
* Surveillance Systems

## Future Improvements

* Support for more weather conditions
* Real-time deployment on edge devices
* Integration with traffic management systems
* Improved nighttime detection
* Multi-camera vehicle tracking

## Author

Abu Yasir | Shafaq Zaman | Kulsoom

## License

This project is licensed under the MIT License.
