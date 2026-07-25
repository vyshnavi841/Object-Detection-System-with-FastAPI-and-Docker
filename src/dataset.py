import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import platform
import subprocess
import time

def download_and_extract(url, data_dir):
    """
    Downloads the dataset zip file from url and extracts it to data_dir.
    Handles flaky connections by trying BITS on Windows and chunked retries elsewhere.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path / "african-wildlife.zip"
    extract_path = data_path / "extracted"

    if extract_path.exists():
        print(f"Dataset already extracted at {extract_path}")
        return extract_path

    # Clean up any failed partial downloads before starting
    if zip_path.exists() and zip_path.stat().st_size < 10000000:
        print("Removing incomplete zip file from previous failed run...")
        os.remove(zip_path)

    if not zip_path.exists():
        print(f"Downloading dataset from {url}...")
        
        # Try BITS Transfer on Windows (extremely robust against network drops)
        download_success = False
        if platform.system() == "Windows":
            print("Detected Windows. Attempting BITS transfer for robust download...")
            try:
                # BITS command via PowerShell
                cmd = f"powershell -Command \"Start-BitsTransfer -Source '{url}' -Destination '{zip_path.as_posix()}'\""
                subprocess.run(cmd, shell=True, check=True)
                download_success = True
                print("BITS transfer download completed successfully.")
            except Exception as e:
                print(f"BITS transfer failed: {e}. Falling back to Python download.")
        
        # Chunked download with retries (fallback / Linux / Docker)
        if not download_success:
            retries = 5
            for attempt in range(retries):
                try:
                    print(f"Download attempt {attempt+1}/{retries}...")
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, 'wb') as out_file:
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                    print("Download complete.")
                    download_success = True
                    break
                except Exception as e:
                    print(f"Download attempt failed: {e}. Retrying in 5 seconds...")
                    if zip_path.exists():
                        try:
                            os.remove(zip_path)
                        except:
                            pass
                    time.sleep(5)
            
            if not download_success:
                raise RuntimeError("Failed to download dataset after multiple attempts.")

    print(f"Extracting dataset to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
    
    # Remove the zip to save space
    if zip_path.exists():
        os.remove(zip_path)
        
    return extract_path

def split_dataset(extracted_dir, data_dir, max_images=None, seed=42):
    """
    Finds all image-label pairs in extracted_dir, deterministically shuffles,
    splits them (70% train, 15% val, 15% test), and structures them into data_dir.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Output paths
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    # If directories already exist and contain images, we skip the split to avoid redone work
    if (train_dir / "images").exists() and len(list((train_dir / "images").glob("*"))) > 0:
        print("Dataset splits already exist in target folders. Skipping split.")
        return

    print("Gathering image-label pairs from extracted directory...")
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []
    
    # We walk the extracted directory to find all images
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                all_images.append(Path(root) / file)

    # Find corresponding label files (same name, but .txt)
    pairs = []
    for img_path in all_images:
        # Search for a label file with the same name in the hierarchy
        # Usually it is in a "labels" directory parallel to "images" or in the same directory
        label_name = img_path.stem + ".txt"
        label_path = None
        
        # Check standard YOLO structure: replace "/images/" with "/labels/"
        possible_label_path = Path(str(img_path.parent).replace("images", "labels")) / label_name
        if possible_label_path.exists():
            label_path = possible_label_path
        else:
            # Search recursively or locally
            local_label = img_path.parent / label_name
            if local_label.exists():
                label_path = local_label
            else:
                # Search in the entire extracted folder (worst case)
                for root, _, files in os.walk(extracted_dir):
                    if label_name in files:
                        label_path = Path(root) / label_name
                        break
        
        if label_path:
            pairs.append((img_path, label_path))

    # Sort pairs alphabetically by image path to make shuffling deterministic across environments
    pairs.sort(key=lambda x: str(x[0]))
    
    print(f"Total image-label pairs found: {len(pairs)}")

    if max_images and max_images < len(pairs):
        pairs = pairs[:max_images]
        print(f"Restricted dataset size to {max_images} images for fast training/evaluation.")

    # Shuffle deterministically
    random.shuffle(pairs)

    # Splits: 70% Train, 15% Val, 15% Test
    n = len(pairs)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train+n_val],
        "test": pairs[n_train+n_val:]
    }

    # Write splits
    for split_name, split_pairs in splits.items():
        split_img_dir = data_path / split_name / "images"
        split_lbl_dir = data_path / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"Copying {len(split_pairs)} pairs to {split_name} split...")
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, split_img_dir / img_path.name)
            shutil.copy2(label_path, split_lbl_dir / label_path.name)

    print("Dataset splitting and structuring completed successfully.")

def get_transforms(img_size, is_train=True):
    """
    Defines Albumentations transform pipelines.
    Includes HorizontalFlip and ShiftScaleRotate for spatial, ColorJitter for color space.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, split, img_size, transform=None):
        """
        Custom PyTorch Dataset for loading images and YOLO format bounding box annotations.
        """
        self.split_dir = Path(data_dir) / split
        self.img_dir = self.split_dir / "images"
        self.label_dir = self.split_dir / "labels"
        self.img_size = img_size
        self.transform = transform

        # Get list of image files
        self.image_files = sorted(list(self.img_dir.glob("*")))
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        bboxes = []
        labels = []

        # Parse labels
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_c, y_c, bbox_w, bbox_h = map(float, parts[1:])
                        
                        # Convert normalized YOLO coordinates to absolute Pascal VOC coordinates [x_min, y_min, x_max, y_max]
                        x_min = (x_c - bbox_w / 2.0) * w
                        y_min = (y_c - bbox_h / 2.0) * h
                        x_max = (x_c + bbox_w / 2.0) * w
                        y_max = (y_c + bbox_h / 2.0) * h
                        
                        # Clip coordinates to image boundary
                        x_min = max(0.0, min(x_min, float(w - 1)))
                        y_min = max(0.0, min(y_min, float(h - 1)))
                        x_max = max(0.0, min(x_max, float(w)))
                        y_max = max(0.0, min(y_max, float(h)))

                        # Avoid invalid boxes (zero width or height)
                        if (x_max > x_min) and (y_max > y_min):
                            bboxes.append([x_min, y_min, x_max, y_max])
                            # For Faster R-CNN, classes must be 1-indexed. (0 is background).
                            # So we add 1 to the class_id (which is 0, 1, 2, 3) -> 1, 2, 3, 4
                            labels.append(class_id + 1)

        # Handle images with no bounding boxes
        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        if self.transform:
            try:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)
                img = transformed['image']
                bboxes = transformed['bboxes']
                labels = transformed['class_labels']
            except Exception as e:
                # If augmentation fails, fall back to basic resize and normalization
                print(f"Warning: Augmentation failed for {img_path.name} with error {e}. Falling back to basic transforms.")
                fallback = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
                transformed = fallback(image=img, bboxes=bboxes, class_labels=labels)
                img = transformed['image']
                bboxes = transformed['bboxes']
                labels = transformed['class_labels']

        # Format target dictionary for PyTorch Faster R-CNN
        target = {}
        if len(bboxes) > 0:
            target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            
        target['iscrowd'] = torch.zeros((len(bboxes),), dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])

        return img, target

def collate_fn(batch):
    """
    Collate function for DataLoader.
    Since target dicts have variable sizes, we zip them as list/tuples.
    """
    return tuple(zip(*batch))
