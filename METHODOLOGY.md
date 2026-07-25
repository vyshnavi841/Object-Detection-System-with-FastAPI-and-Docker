# Technical Methodology & Architectural Rationale

This document provides a comprehensive technical breakdown of the engineering decisions, model architecture trade-offs, dataset preprocessing methodologies, evaluation metrics, and containerized deployment practices implemented in this Custom Object Detection system.

---

## 1. Dataset Selection & Preprocessing Strategy

### 1.1 Dataset Choice & Characterization
For this project, the **African Wildlife Dataset** was selected. It consists of 500+ high-resolution natural scene images containing four distinct target animal classes:
1. `buffalo`
2. `elephant`
3. `rhino`
4. `zebra`

#### Selection Rationale:
- **Class Variance & Real-World Complexity**: The dataset features objects in unconstrained, natural safari environments with varying scale, occlusion, clutter, and lighting conditions.
- **Balanced Class Distribution**: Each class contains balanced instances, avoiding extreme class imbalance while providing enough intra-class variance (different poses, group herds, close-up vs distant shots).
- **Feasible Compute Scale**: The dataset size allows fast, reproducible fine-tuning and evaluation on standard hardware (CPU/GPU) without requiring multi-GPU clusters.

---

### 1.2 Annotation Parsing & Uniform Internal Representation
Bounding box annotations in standard object detection benchmarks appear in different formats:
- **COCO**: JSON format with absolute coordinates `[x_min, y_min, width, height]`.
- **Pascal VOC**: XML format with absolute coordinates `[x_min, y_min, x_max, y_max]`.
- **YOLO**: Normalized relative coordinates `[x_center, y_center, width, height]` (values between 0.0 and 1.0).

In `src/dataset.py`, annotations are parsed from normalized YOLO format into absolute Pascal VOC coordinates `[x_min, y_min, x_max, y_max]` using image dimensions \((w, h)\):

\[
x_{\text{min}} = \max\left(0, \left(x_{\text{center}} - \frac{\text{width}}{2}\right) \times w\right)
\]
\[
y_{\text{min}} = \max\left(0, \left(y_{\text{center}} - \frac{\text{height}}{2}\right) \times h\right)
\]
\[
x_{\text{max}} = \min\left(w, \left(x_{\text{center}} + \frac{\text{width}}{2}\right) \times w\right)
\]
\[
y_{\text{max}} = \min\left(h, \left(y_{\text{center}} + \frac{\text{height}}{2}\right) \times h\right)
\]

Coordinates are strictly clamped to image boundaries \([0, w]\) and \([0, h]\) to prevent out-of-bounds indexing or negative box dimensions.

---

### 1.3 Deterministic Dataset Splitting
To strictly eliminate **data leakage** (where test or validation images contaminate training updates), dataset splitting is performed deterministically:
- **Split Ratio**: 70% Training, 15% Validation, 15% Testing.
- **Deterministic Seeding**: Shuffling is governed by `random.seed(42)` and `np.random.seed(42)`. Image-label file pairs are first sorted alphabetically by path before shuffling to ensure exact, cross-platform reproducibility across environments.

---

### 1.4 Bounding-Box-Aware Data Augmentation
Applying standard image transformations (like cropping, rotation, or horizontal flipping) without updating bounding box coordinates corrupts training: the model learns to associate target class labels with wrong image regions.

We utilize **Albumentations** with explicit `BboxParams(format='pascal_voc', label_fields=['class_labels'])`:

1. **Spatial Transformations**:
   - `A.HorizontalFlip(p=0.5)`: Mirrors pixel columns \(x' = w - x\) and swaps \(x_{\text{min}}\) with \(x_{\text{max}}\).
   - `A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)`: Rotates and rescales bounding box vertices using 2D affine transformation matrices.

2. **Color Space Transformations**:
   - `A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)`: Alters HSV/RGB channels without modifying bounding box geometry.

3. **Normalization & Tensor Conversion**:
   - Standard ImageNet channel normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`) converted to PyTorch `torch.Tensor` `(C, H, W)`.

---

## 2. Model Architecture Selection & Transfer Learning

### 2.1 Comparative Architecture Evaluation

| Model Architecture | Parameter Count | Weight File Size | Primary Advantage | Limitation / Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **Faster R-CNN (ResNet50 FPN)** | ~41.5M | ~160 MB | Highest detection precision | High CPU latency (~400ms/img), large container size |
| **YOLOv8 Small (Ultralytics)** | ~11.2M | ~22 MB | Extremely fast single-stage detection | Requires additional GPL license dependencies |
| **SSDLite MobileNetV3 Large (Selected)** | **~3.2M** | **~9.3 MB** | **Ultra-lightweight, ultra-fast CPU inference (<50ms)** | Slightly lower recall on ultra-small objects |

### 2.2 Why SSDLite MobileNetV3 Large?
1. **Lightweight Deployment Footprint**: The model weights file is only **9.3 MB**, allowing extremely fast Docker image layer caching and minimal container memory consumption (~200MB RAM usage at runtime).
2. **Real-time CPU Inference**: MobileNetV3 uses Depthwise Separable Convolutions and SSDLite detection heads, making it optimized for CPU container execution without requiring expensive GPU infrastructure.
3. **Pre-trained COCO Backbone Transfer Learning**: The feature extractor retains foundational representations (edge detectors, texture filters, shape descriptors) pre-trained on Microsoft COCO (80 classes, 300k+ images). Transfer learning fine-tunes only the custom classification and regression heads for the 4 wildlife target classes.

---

## 3. Evaluation Metrics & Performance Analysis

### 3.1 Intersection over Union (IoU)
IoU measures the spatial overlap accuracy between predicted box \(B_p\) and ground truth box \(B_{gt}\):

\[
\text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}
\]

Predictions are classified as:
- **True Positive (TP)**: \(\text{IoU} \ge \text{threshold}\) and predicted class matches ground truth label.
- **False Positive (FP)**: \(\text{IoU} < \text{threshold}\) or duplicate detection for an already matched ground truth box.
- **False Negative (FN)**: Ground truth object missed by predictions.

---

### 3.2 Mean Average Precision (mAP) Calculation
1. **Precision & Recall Curves**: Predictions are sorted by confidence score in descending order. Cumulative Precision \(P = \frac{TP}{TP + FP}\) and Recall \(R = \frac{TP}{TP + FN}\) are computed.
2. **Average Precision (AP)**: Computed via all-point interpolation of the Precision-Recall curve:
   \[
   \text{AP} = \int_{0}^{1} P_{\text{interp}}(R) \, dR
   \]
3. **mAP Metrics**:
   - **mAP@0.5**: Mean AP across all 4 classes at an IoU threshold of 0.50.
   - **mAP@0.5:0.95**: Mean AP averaged over 10 IoU thresholds from 0.50 to 0.95 with step 0.05.

---

### 3.3 Evaluation Results Analysis
In our test set evaluations on the African Wildlife dataset:
- **mAP@0.5**: Strong localization performance on distinct, large-scale instances (e.g., `elephant` and `zebra` due to high contrast patterns and distinct silhouettes).
- **mAP@0.5:0.95**: Higher IoU thresholds (0.80+) penalize minor bounding box boundary shifts when animals are partially occluded by tall grass or trees.

---

## 4. MLOps & Production Containerization

### 4.1 REST API Design (FastAPI)
The inference service is exposed via an asynchronous REST API (`src/api.py`):
- **Endpoint**: `POST /predict` (`multipart/form-data`)
- **Real-Time Preprocessing**: Binary bytes are decoded into OpenCV matrix (`cv2.imdecode`), converted to RGB, resized to `320x320`, normalized, and passed to PyTorch inference engine in single-batch format.
- **Post-processing & NMS**: Non-Maximum Suppression (NMS) filters overlapping boxes, rescales bounding box coordinates back to original image resolution `(w_orig, h_orig)`, and formats the JSON payload strictly according to the contract:

```json
{
  "predictions": [
    {
      "class": "string",
      "confidence": 0.95,
      "bbox": [10.0, 20.0, 300.0, 400.0]
    }
  ]
}
```

### 4.2 Docker Optimization Strategy
- **Base Image**: `python:3.11-slim` minimizes total image size compared to standard Python images.
- **Headless OpenCV**: `opencv-python-headless` eliminates GUI window dependencies (`libX11`, `libQt`).
- **Pre-packaged Model Weights**: `best_model.pth` is explicitly copied during Docker build, guaranteeing immediate model readiness upon container startup without cold-start download delays.
