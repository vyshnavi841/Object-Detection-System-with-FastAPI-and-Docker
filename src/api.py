import os
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Setup FastAPI App
app = FastAPI(
    title="Custom Object Detection API",
    description="Inference API for detecting African Wildlife (Buffalo, Elephant, Rhino, Zebra) using SSDLite MobileNetV3.",
    version="1.0"
)

# Load Configuration
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    # Fallback default configuration if config.yaml is not present
    config = {
        'dataset': {
            'img_size': 320,
            'num_classes': 4,
            'classes': ["buffalo", "elephant", "rhino", "zebra"]
        },
        'training': {
            'device': 'cpu'
        },
        'inference': {
            'confidence_threshold': 0.5
        }
    }
else:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

img_size = config['dataset']['img_size']
num_classes = config['dataset']['num_classes']
classes = config['dataset']['classes']
conf_threshold = config.get('inference', {}).get('confidence_threshold', 0.5)

from evaluate import get_model

# Setup Model
device = torch.device('cpu') # Always default to CPU for API container inference unless specified otherwise
model = get_model(num_classes, weights_name="NONE")

weights_path = Path("best_model.pth")
if weights_path.exists():
    print(f"Loading trained weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print(f"Warning: No weights found at {weights_path}. Model will return random detections.")

model.to(device)
model.eval()

# Pydantic Schemas for Response Validation
class DetectionResult(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float
    bbox: List[float] = Field(..., description="[x_min, y_min, x_max, y_max]")

    class Config:
        populate_by_name = True

class PredictResponse(BaseModel):
    predictions: List[DetectionResult]

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "model": "Faster R-CNN MobileNetV3 Large 320 FPN",
        "classes": classes,
        "weights_loaded": weights_path.exists()
    }

@app.post("/predict", response_model=PredictResponse, status_code=200)
async def predict(file: UploadFile = File(...)):
    # Validate uploaded file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        # Read uploaded image bytes
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

        h_orig, w_orig, _ = img.shape

        # Preprocessing: convert to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))

        # Normalize and convert to PyTorch Tensor
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
        
        # Transpose to (C, H, W) and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            output = outputs[0]

        pred_boxes = output['boxes'].cpu().numpy()
        pred_labels = output['labels'].cpu().numpy()
        pred_scores = output['scores'].cpu().numpy()

        # Coordinate scaling factors (back to original size)
        scale_x = w_orig / img_size
        scale_y = h_orig / img_size

        response_detections = []
        
        # Format predictions above threshold
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score >= conf_threshold:
                # Class 0 in model corresponds to background, target class name list starts from 1-indexed labels
                class_idx = label - 1
                if 0 <= class_idx < len(classes):
                    class_name = classes[class_idx]
                else:
                    class_name = f"unknown_{label}"
                
                # Scale coordinates back to original size
                x_min = float(box[0] * scale_x)
                y_min = float(box[1] * scale_y)
                x_max = float(box[2] * scale_x)
                y_max = float(box[3] * scale_y)

                # Ensure boundaries match original image dimensions
                x_min = max(0.0, min(x_min, float(w_orig - 1)))
                y_min = max(0.0, min(y_min, float(h_orig - 1)))
                x_max = max(0.0, min(x_max, float(w_orig)))
                y_max = max(0.0, min(y_max, float(h_orig)))

                response_detections.append(
                    DetectionResult(
                        class_name=class_name,
                        confidence=float(score),
                        bbox=[round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)]
                    )
                )

        return PredictResponse(predictions=response_detections)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
