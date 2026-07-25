import os
import sys
import argparse
import time
from pathlib import Path

# Ensure src directory is in sys.path for internal module imports
src_dir = str(Path(__file__).parent.resolve())
if src_dir not in sys.path:
    sys.path.append(src_dir)

import yaml
import torch
from torch.utils.data import DataLoader
from dataset import (
    download_and_extract,
    split_dataset,
    get_transforms,
    ObjectDetectionDataset,
    collate_fn
)
from evaluate import get_model, evaluate_map

def run_training(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    dataset_url = config['dataset']['zip_url']
    data_dir = config['dataset']['data_dir']
    img_size = config['dataset']['img_size']
    num_classes = config['dataset']['num_classes']
    classes = config['dataset']['classes']
    max_images = config['dataset'].get('max_images', None)
    
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    opt_name = config['training']['optimizer']
    weight_decay = config['training']['weight_decay']
    num_workers = config['training']['num_workers']
    device_name = config['training'].get('device', 'cpu')

    device = torch.device('cuda' if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Using training device: {device}")

    # Phase 1: Download and split dataset
    print("\n--- PHASE 1: DATASET ACQUISITION AND SPLITTING ---")
    extracted_path = download_and_extract(dataset_url, data_dir)
    split_dataset(extracted_path, data_dir, max_images=max_images, seed=42)

    # Phase 2: Create DataLoaders
    print("\n--- PHASE 2: CREATING DATALOADERS AND AUGMENTATIONS ---")
    train_transform = get_transforms(img_size, is_train=True)
    val_transform = get_transforms(img_size, is_train=False)

    train_dataset = ObjectDetectionDataset(data_dir, "train", img_size, transform=train_transform)
    val_dataset = ObjectDetectionDataset(data_dir, "val", img_size, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Phase 3: Model and Optimizer Initialization
    print("\n--- PHASE 3: MODEL INITIALIZATION AND TRANSFER LEARNING ---")
    model = get_model(num_classes)
    model.to(device)

    # Select Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Phase 4: Training Loop
    print("\n--- PHASE 4: TRAINING PIPELINE ---")
    best_val_map = -1.0
    training_log_path = Path("training.log")
    
    # Clean previous log file
    if training_log_path.exists():
        os.remove(training_log_path)

    with open(training_log_path, 'a') as log_file:
        header = "Epoch,Train_Loss,Val_mAP_50\n"
        log_file.write(header)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass (Faster R-CNN returns dictionary of losses in training mode)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass & optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {losses.item():.4f} - Speed: {elapsed/(batch_idx+1):.2f}s/batch")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} complete. Average Train Loss: {avg_loss:.4f}")

        # Evaluation on Validation set to check validation mAP
        print(f"Evaluating validation set...")
        model.eval()
        val_preds = []
        val_gts = []

        with torch.no_grad():
            for images, targets in val_loader:
                images_device = [img.to(device) for img in images]
                outputs = model(images_device)

                for target, output in zip(targets, outputs):
                    img_id = target['image_id'].item()
                    
                    val_gts.append({
                        'image_id': img_id,
                        'boxes': target['boxes'].cpu().numpy().tolist(),
                        'labels': target['labels'].cpu().numpy().tolist()
                    })

                    val_preds.append({
                        'image_id': img_id,
                        'boxes': output['boxes'].cpu().numpy().tolist(),
                        'labels': output['labels'].cpu().numpy().tolist(),
                        'scores': output['scores'].cpu().numpy().tolist()
                    })

        val_map50 = evaluate_map(val_preds, val_gts, num_classes, iou_threshold=0.5)
        print(f"Validation mAP@0.5: {val_map50:.4f}")

        # Log metrics
        with open(training_log_path, 'a') as log_file:
            log_file.write(f"{epoch},{avg_loss:.4f},{val_map50:.4f}\n")

        # Save best weights
        if val_map50 > best_val_map:
            best_val_map = val_map50
            print(f"New best validation mAP! Saving weights to best_model.pth...")
            torch.save(model.state_dict(), "best_model.pth")

    print("\nTraining completed!")
    print(f"Best Validation mAP@0.5: {best_val_map:.4f}")
    print(f"Weights saved at: best_model.pth")
    print(f"Metrics logged at: {training_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_training(args.config)
