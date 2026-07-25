# Custom Object Detection System with FastAPI and Docker

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

A production-ready, end-to-end computer vision and MLOps pipeline for custom object detection. This project demonstrates fine-tuning a state-of-the-art deep neural network (**SSDLite MobileNetV3 Large**) on a multi-class dataset (**African Wildlife**: Buffalo, Elephant, Rhino, Zebra), performing deterministic dataset splits, bounding-box-aware data augmentations via **Albumentations**, evaluating detection performance using **mAP@0.5** and **mAP@0.5:0.95**, and serving real-time predictions via a **FastAPI REST API** containerized with **Docker**.

---

## 📽️ Video Demonstration

Watch the complete 5-8 minute technical walkthrough covering codebase architecture, dataset pipeline, training/evaluation execution, and live Docker API inferencing:

👉 **[Watch Project Walkthrough Video](https://www.youtube.com/watch?v=placeholder_video_id)**

---

## 🏗️ Project Architecture

The system consists of two decoupled but integrated pipelines:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                    TRAINING PIPELINE                    │
                  └─────────────────────────────────────────────────────────┘
                                               │
 ┌──────────────────────┐             ┌─────────────────┐             ┌─────────────────────┐
 │ Raw Dataset (Zip/URL)│ ──────────► │  Parse & Split  │ ──────────► │ Albumentations Augs │
 └──────────────────────┘             │ (70/15/15 Seed) │             │ (Spatial + Color)   │
                                      └─────────────────┘             └─────────────────────┘
                                                                                 │
 ┌──────────────────────┐             ┌─────────────────┐             ┌─────────────────────┐
 │ Pre-trained Weights  │ ──────────► │ Transfer Learn  │ ◄───────────│   PyTorch Loader    │
 │ (COCO Backbone)      │             │ Training Loop   │             │   (Batching)        │
 └──────────────────────┘             └─────────────────┘             └─────────────────────┘
                                               │
                                               ▼
                                     ┌──────────────────┐             ┌─────────────────────┐
                                     │ best_model.pth   │ ──────────► │ Evaluation Script   │
                                     │ (Trained Weights)│             │ (mAP & Visuals)     │
                                     └──────────────────┘             └─────────────────────┘
                                               │
                  ┌────────────────────────────┴────────────────────────────┐
                  │                   INFERENCE PIPELINE                    │
                  └─────────────────────────────────────────────────────────┘
                                               │
 ┌──────────────────────┐             ┌─────────────────┐             ┌─────────────────────┐
 │ HTTP POST /predict   │ ──────────► │ FastAPI Server  │ ──────────► │ cv2 Image Decode &  │
 │ (Image File Payload) │             │ (Uvicorn Async) │             │ Tensor Normalization│
 └──────────────────────┘             └─────────────────┘             └─────────────────────┘
                                                                                 │
 ┌──────────────────────┐             ┌─────────────────┐             ┌─────────────────────┐
 │ Structured JSON Output│ ◄────────── │ Post-processing │ ◄───────────│ Model Forward Pass  │
 │ 200 OK Response      │             │ (NMS & Scaling) │             │ (CPU / GPU)         │
 └──────────────────────┘             └─────────────────┘             └─────────────────────┘
```

---

## 📁 Repository Structure

```
project-root/
├── data/                        # Excluded from git; created dynamically at runtime
│   ├── train/                   # 70% Training images and YOLO annotations
│   ├── val/                     # 15% Validation images and YOLO annotations
│   └── test/                    # 15% Test images and YOLO annotations
├── src/
│   ├── dataset.py               # Dataset acquisition, parsing, deterministic splitting, and Albumentations
│   ├── train.py                 # Fine-tuning training loop, metric logging, and model checkpointing
│   ├── evaluate.py              # mAP@0.5 / mAP@0.5:0.95 evaluation and prediction visualization
│   └── api.py                   # FastAPI REST API endpoints and Pydantic response formatting
├── output/
│   └── predictions/             # Annotated test set prediction images with bounding boxes & scores
├── config.yaml                  # Centralized hyperparameter and dataset configuration
├── submission.yml               # Automated evaluation command definitions (train, evaluate, deploy)
├── Dockerfile                   # Multi-stage optimized Docker build definition
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Setup, execution, and API documentation
└── METHODOLOGY.md               # Detailed engineering rationale and architectural design decisions
```

---

## ⚙️ Configuration (`config.yaml`)

All hyperparameters, dataset parameters, and inference settings are centralized in `config.yaml`:

```yaml
dataset:
  name: "african-wildlife"
  zip_url: "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
  data_dir: "data"
  img_size: 320
  num_classes: 4
  classes: ["buffalo", "elephant", "rhino", "zebra"]
  max_images: 500  # Set to null for full dataset execution

training:
  batch_size: 8
  epochs: 3
  learning_rate: 0.001
  optimizer: "adam"
  weight_decay: 0.0005
  num_workers: 0
  device: "cpu"  # Automatically uses CUDA if available and requested

inference:
  confidence_threshold: 0.5
  iou_threshold: 0.5
```

---

## 🚀 Quick Start & Local Execution

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training Pipeline
The training script automatically downloads the raw dataset zip, splits it deterministically with a fixed seed (`seed=42`), initializes the pre-trained SSDLite MobileNetV3 model, and executes the training loop while logging validation mAP to `training.log`:

```bash
python src/train.py --config config.yaml
```

### 4. Run Evaluation & Generate Visual Predictions
The evaluation script loads `best_model.pth`, computes **mAP@0.5** and **mAP@0.5:0.95** over the unseen test set, and saves annotated visual prediction images to `output/predictions/`:

```bash
python src/evaluate.py --config config.yaml
```

### 5. Launch FastAPI Local Server
To run the REST API locally with Uvicorn:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🐳 Docker Deployment

The application is fully containerized using an optimized `python:3.11-slim` base image with pre-installed OpenCV runtime dependencies and pre-loaded PyTorch model weights (`best_model.pth`).

### Build Docker Image
```bash
docker build -t object-detector .
```

### Run Docker Container
```bash
docker run -d -p 8000:8000 --name od-container object-detector
```

The container exposes port `8000` and immediately starts serving real-time inference requests.

---

## 🤖 Automated Execution (`submission.yml`)

The repository includes a standardized `submission.yml` file for automated verification:

```yaml
version: "1.0"
commands:
  train: "python src/train.py --config config.yaml"
  evaluate: "python src/evaluate.py --config config.yaml"
  deploy: |
    docker build -t object-detector .
    docker run -d -p 8000:8000 --name od-container object-detector
```

---

## 📡 REST API Documentation

### Endpoints

| Method | Endpoint | Description | Payload | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | API Health Check & System Status | None | JSON with health status and model details |
| `POST` | `/predict` | Real-time Object Detection Inference | `multipart/form-data` (`file`) | JSON array of detected objects |

---

### Request & Response Specifications

#### `POST /predict`

**Headers:**
`Content-Type: multipart/form-data`

**Body:**
`file`: Binary image file (JPEG, PNG, BMP)

**Response Schema (`HTTP 200 OK`):**
```json
{
  "predictions": [
    {
      "class": "zebra",
      "confidence": 0.94,
      "bbox": [120.5, 45.0, 200.0, 130.5]
    },
    {
      "class": "elephant",
      "confidence": 0.88,
      "bbox": [210.0, 80.2, 450.0, 310.0]
    }
  ]
}
```
*(Note: Bounding boxes follow `[x_min, y_min, x_max, y_max]` in absolute image pixel coordinates.)*

---

### Example API Usage

#### Querying with `curl`:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_image.jpg'
```

#### Querying with Python `requests`:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("sample_image.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

---

## 🧪 License & Attribution
Developed as part of the Custom Object Detection & MLOps System Project. Built using PyTorch, torchvision, Albumentations, FastAPI, OpenCV, and Docker.
