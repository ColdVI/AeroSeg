#!/usr/bin/env python3
"""
Organize Semantic Drone Dataset into train/val/test splits.
"""

import os
import shutil
import random
from pathlib import Path

# Configuration
DATA_DIR = Path("/Users/anil/Desktop/AntiGravity/AI-Project/data")
RAW_DIR = DATA_DIR / "raw"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def organize_dataset():
    print("Organizing Semantic Drone Dataset...")
    
    # Actual dataset paths
    image_dir = RAW_DIR / "dataset" / "semantic_drone_dataset" / "original_images"
    mask_dir = RAW_DIR / "dataset" / "semantic_drone_dataset" / "label_images_semantic"
    
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    
    # Get all images
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"Found {len(images)} images")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(images)
    
    n_train = int(len(images) * TRAIN_RATIO)
    n_val = int(len(images) * VAL_RATIO)
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Copy files
    for split, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        img_dest = DATA_DIR / split / "images"
        mask_dest = DATA_DIR / split / "masks"
        
        for img_path in split_images:
            # Copy image
            shutil.copy(img_path, img_dest / img_path.name)
            
            # Find and copy corresponding mask
            mask_name = img_path.stem + ".png"
            mask_path = mask_dir / mask_name
            
            if mask_path.exists():
                shutil.copy(mask_path, mask_dest / mask_name)
            else:
                # Try alternative mask naming
                for alt_mask in mask_dir.glob(f"*{img_path.stem}*"):
                    shutil.copy(alt_mask, mask_dest / alt_mask.name)
                    break
        
        print(f"[✓] {split}: {len(list(img_dest.glob('*')))} images, {len(list(mask_dest.glob('*')))} masks")

if __name__ == "__main__":
    organize_dataset()
    print("\nDone! You can now run: python3 train.py --data ./data")
