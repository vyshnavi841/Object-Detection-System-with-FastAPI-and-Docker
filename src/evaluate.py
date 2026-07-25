import os
import random
import argparse
from pathlib import Path
import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torch.utils.data import DataLoader
from dataset import ObjectDetectionDataset, get_transforms, collate_fn

# Define a color palette for visual boxes (B, G, R)
COLOR_PALETTE = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0)   # Cyan
]

def calculate_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x_min, y_min, x_max, y_max].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
        
    return inter_area / union_area

def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) under the Precision-Recall curve using COCO/VOC 11-point or all-points style.
    We implement all-points interpolation (VOC 2012 / COCO style).
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Compute AP as sum of areas
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """
    Computes mAP for a given IoU threshold.
    Args:
        predictions: List of dicts, each containing:
            'image_id': int
            'boxes': list of [x_min, y_min, x_max, y_max]
            'labels': list of int (1-indexed)
            'scores': list of float
        ground_truths: List of dicts, each containing:
            'image_id': int
            'boxes': list of [x_min, y_min, x_max, y_max]
            'labels': list of int (1-indexed)
        num_classes: Number of target classes (excluding background).
        iou_threshold: Float IoU threshold.
    """
    ap_scores = []
    
    # We iterate through all classes (1 to num_classes)
    for c in range(1, num_classes + 1):
        # Gather all ground truths for class c
        class_gts = {}
        total_gts = 0
        for gt in ground_truths:
            img_id = gt['image_id']
            class_boxes = [box for box, lbl in zip(gt['boxes'], gt['labels']) if lbl == c]
            class_gts[img_id] = {
                'boxes': class_boxes,
                'matched': [False] * len(class_boxes)
            }
            total_gts += len(class_boxes)

        # Gather all predictions for class c
        class_preds = []
        for pred in predictions:
            img_id = pred['image_id']
            for box, lbl, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                if lbl == c:
                    class_preds.append({
                        'image_id': img_id,
                        'box': box,
                        'score': score
                    })

        # If there are no ground truths for this class, skip it
        if total_gts == 0:
            continue

        # If there are no predictions, AP is 0
        if len(class_preds) == 0:
            ap_scores.append(0.0)
            continue

        # Sort predictions by score descending
        class_preds.sort(key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for idx, pred in enumerate(class_preds):
            img_id = pred['image_id']
            pred_box = pred['box']
            
            # Find matching ground truths in the same image
            gt_info = class_gts.get(img_id)
            if gt_info is None or len(gt_info['boxes']) == 0:
                fp[idx] = 1
                continue

            # Compute IoU with all GT boxes in the same image
            best_iou = -1.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_info['boxes']):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                if not gt_info['matched'][best_gt_idx]:
                    tp[idx] = 1
                    gt_info['matched'][best_gt_idx] = True
                else:
                    fp[idx] = 1  # Double detection
            else:
                fp[idx] = 1

        # Compute cumulative precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        precisions = cum_tp / (cum_tp + cum_fp + 1e-16)
        recalls = cum_tp / (total_gts + 1e-16)

        ap = compute_ap(recalls, precisions)
        ap_scores.append(ap)

    if len(ap_scores) == 0:
        return 0.0
        
    return np.mean(ap_scores)

def compute_map_coco(predictions, ground_truths, num_classes):
    """
    Computes mAP@0.5 and mAP@0.5:0.95.
    """
    # Compute mAP@0.5
    map50 = evaluate_map(predictions, ground_truths, num_classes, iou_threshold=0.5)
    
    # Compute mAP@0.5:0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps = []
    for iou_thresh in iou_thresholds:
        m = evaluate_map(predictions, ground_truths, num_classes, iou_threshold=iou_thresh)
        maps.append(m)
        
    map50_95 = np.mean(maps)
    
    return map50, map50_95

def get_model(num_classes, weights_name="DEFAULT"):
    """
    Initializes an SSDLite MobileNetV3 model and swaps the classification head for custom classes.
    """
    if weights_name == "DEFAULT":
        weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    else:
        weights = None
        
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    in_channels = [m[0][0].in_channels for m in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # swapping classification head for target classes
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
        norm_layer=nn.BatchNorm2d
    )
    return model

def run_evaluation(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['dataset']['data_dir']
    img_size = config['dataset']['img_size']
    num_classes = config['dataset']['num_classes']
    classes = config['dataset']['classes']
    device_name = config['training'].get('device', 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')

    print(f"Running evaluation on device: {device}")

    # Load test dataset
    transform = get_transforms(img_size, is_train=False)
    test_dataset = ObjectDetectionDataset(data_dir, "test", img_size, transform=transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    # Initialize model and load weights
    model = get_model(num_classes)
    weights_path = Path("best_model.pth")
    if not weights_path.exists():
        print(f"Warning: {weights_path} not found! Initializing model with default pre-trained weights for dummy evaluation.")
    else:
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.to(device)
    model.eval()

    all_preds = []
    all_gts = []

    # Run inference over the test dataset
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs_device = [img.to(device) for img in imgs]
            outputs = model(imgs_device)

            for idx, (target, output) in enumerate(zip(targets, outputs)):
                img_id = target['image_id'].item()
                
                # Ground truths
                gt_boxes = target['boxes'].cpu().numpy().tolist()
                gt_labels = target['labels'].cpu().numpy().tolist()
                
                all_gts.append({
                    'image_id': img_id,
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })

                # Model predictions (NMS is already applied internally by torchvision Faster R-CNN)
                pred_boxes = output['boxes'].cpu().numpy().tolist()
                pred_labels = output['labels'].cpu().numpy().tolist()
                pred_scores = output['scores'].cpu().numpy().tolist()

                all_preds.append({
                    'image_id': img_id,
                    'boxes': pred_boxes,
                    'labels': pred_labels,
                    'scores': pred_scores
                })

    # Calculate metrics
    map50, map50_95 = compute_map_coco(all_preds, all_gts, num_classes)
    
    print("\n" + "="*40)
    print("TEST EVALUATION RESULTS:")
    print(f"mAP@0.5:        {map50:.4f}")
    print(f"mAP@0.5:0.95:   {map50_95:.4f}")
    print("="*40 + "\n")

    # Generate visual predictions
    visualize_predictions(model, test_dataset, classes, num_classes, device)

    return map50, map50_95

def visualize_predictions(model, dataset, classes, num_classes, device, num_images=5):
    """
    Selects random test images, runs inference, draws boxes/labels/scores,
    and saves the annotated images to output/predictions/.
    """
    output_dir = Path("output/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))
    # We want to keep it deterministic or choose at random
    random.seed(42)
    selected_indices = random.sample(indices, min(num_images, len(dataset)))
    
    model.eval()
    
    with torch.no_grad():
        for idx in selected_indices:
            # We get the raw image file path to draw on the original resolution image
            img_path = dataset.image_files[idx]
            orig_img = cv2.imread(str(img_path))
            h_orig, w_orig, _ = orig_img.shape

            # Prepare tensor for inference
            img_tensor, target = dataset[idx]
            img_device = img_tensor.unsqueeze(0).to(device)

            outputs = model(img_device)
            output = outputs[0]

            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()

            # Bounding box coordinate scaling factor (since the model was run on normalized/resized image_size)
            scale_x = w_orig / dataset.img_size
            scale_y = h_orig / dataset.img_size

            # Draw predicted boxes
            box_count = 0
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score < 0.5:  # Confidence threshold
                    continue
                
                box_count += 1
                # Scale coordinates back to original size
                x_min = int(box[0] * scale_x)
                y_min = int(box[1] * scale_y)
                x_max = int(box[2] * scale_x)
                y_max = int(box[3] * scale_y)

                class_idx = label - 1  # Convert back from 1-indexed to 0-indexed class list
                class_name = classes[class_idx] if class_idx < len(classes) else f"Class {label}"
                color = COLOR_PALETTE[class_idx % len(COLOR_PALETTE)]

                # Draw box
                cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), color, 3)

                # Draw label
                label_text = f"{class_name}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background for text
                cv2.rectangle(
                    orig_img, 
                    (x_min, y_min - text_height - 10), 
                    (x_min + text_width, y_min), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    orig_img, 
                    label_text, 
                    (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )

            # Draw ground truths in white dotted rectangles for qualitative comparison
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            for box, label in zip(gt_boxes, gt_labels):
                x_min = int(box[0] * scale_x)
                y_min = int(box[1] * scale_y)
                x_max = int(box[2] * scale_x)
                y_max = int(box[3] * scale_y)
                
                class_idx = label - 1
                class_name = classes[class_idx] if class_idx < len(classes) else f"Class {label}"
                
                # Draw dashed / thin rectangle for ground truth
                cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
                cv2.putText(
                    orig_img, 
                    f"GT: {class_name}", 
                    (x_min, y_max + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1
                )

            # Save image
            save_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(save_path), orig_img)
            print(f"Saved visualization: {save_path} (Detected {box_count} objects)")

    print(f"Visualization complete. Predicted test set samples saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_evaluation(args.config)
