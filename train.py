#!/usr/bin/env python3
"""
AeroSeg Training Script
=======================

Fine-tune DeepLabV3-MobileNetV3 on aerial segmentation datasets
for improved UAV landing zone detection.

Usage:
    python train.py --data ./data --epochs 50
    python train.py --data ./data --resume checkpoints/best.pth

Supported Datasets:
    - Semantic Drone Dataset (20 classes)
    - UAVid (8 classes)
"""

import os
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =============================================================================
# SEMANTIC DRONE DATASET CLASS MAPPING
# =============================================================================
# Indices from the dataset (0-23) → 3 AeroSeg categories
# Reference: class_dict_seg.csv

# Class index to AeroSeg category mapping
# Category: 0=SAFE, 1=HAZARD, 2=WATER
INDEX_TO_CATEGORY = {
    0: 0,   # unlabeled → SAFE (background)
    1: 0,   # paved-area → SAFE
    2: 0,   # dirt → SAFE
    3: 0,   # grass → SAFE
    4: 0,   # gravel → SAFE
    5: 2,   # water → WATER
    6: 0,   # rocks → SAFE
    7: 2,   # pool → WATER
    8: 1,   # vegetation → HAZARD (dense, not landable)
    9: 1,   # roof → HAZARD
    10: 1,  # wall → HAZARD
    11: 1,  # window → HAZARD
    12: 1,  # door → HAZARD
    13: 1,  # fence → HAZARD
    14: 1,  # fence-pole → HAZARD
    15: 1,  # person → HAZARD
    16: 1,  # dog → HAZARD
    17: 1,  # car → HAZARD
    18: 1,  # bicycle → HAZARD
    19: 1,  # tree → HAZARD (KEY!)
    20: 1,  # bald-tree → HAZARD
    21: 1,  # ar-marker → HAZARD
    22: 1,  # obstacle → HAZARD
    23: 1,  # conflicting → HAZARD
}


# =============================================================================
# DATASET CLASSES
# =============================================================================

class SemanticDroneDataset(Dataset):
    """
    PyTorch Dataset for Semantic Drone Dataset.
    
    Expected structure:
        data/
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── val/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        self.image_dir = self.root_dir / split / 'images'
        self.mask_dir = self.root_dir / split / 'masks'
        
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        self.images = sorted(list(self.image_dir.glob('*.jpg')) + 
                            list(self.image_dir.glob('*.png')))
        
        print(f"[Dataset] Loaded {len(self.images)} images for {split} split")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (grayscale indexed)
        mask_name = img_path.stem + '.png'
        mask_path = self.mask_dir / mask_name
        
        if mask_path.exists():
            mask = Image.open(mask_path)  # Keep as grayscale/palette
        else:
            # Try alternative naming
            mask_path = self.mask_dir / img_path.name.replace('.jpg', '.png')
            mask = Image.open(mask_path) if mask_path.exists() else None
        
        # Resize
        image = image.resize(self.target_size, Image.BILINEAR)
        if mask:
            mask = mask.resize(self.target_size, Image.NEAREST)
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        
        if mask:
            # Convert indexed mask to category labels
            mask_np = np.array(mask)
            category_mask = self._index_to_category(mask_np)
            mask_tensor = torch.from_numpy(category_mask).long()
        else:
            mask_tensor = torch.zeros(self.target_size, dtype=torch.long)
        
        return image, mask_tensor
    
    def _index_to_category(self, index_mask: np.ndarray) -> np.ndarray:
        """Convert indexed mask to category labels (0=SAFE, 1=HAZARD, 2=WATER)."""
        category_mask = np.zeros_like(index_mask, dtype=np.uint8)
        
        for class_idx, category in INDEX_TO_CATEGORY.items():
            category_mask[index_mask == class_idx] = category
        
        return category_mask


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_model(
    num_classes: int = 3,
    model_type: str = 'mobilenet',
    pretrained: bool = True
) -> nn.Module:
    """
    Create segmentation model for training.
    
    Args:
        num_classes: Number of output classes (default: 3)
        model_type: 'mobilenet', 'unet', or 'light'
        pretrained: Use pretrained weights (mobilenet only)
    
    Returns:
        PyTorch model
    """
    if model_type == 'unet':
        from custom_model import AeroSegUNet
        model = AeroSegUNet(num_classes=num_classes)
        print(f"[Train] Created custom U-Net ({model.count_parameters():,} params)")
        
    elif model_type == 'light':
        from custom_model import AeroSegLightUNet
        model = AeroSegLightUNet(num_classes=num_classes)
        print(f"[Train] Created Light U-Net ({model.count_parameters():,} params)")
        
    elif model_type == 'mobilenet':
        if pretrained:
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
        else:
            model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
        
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier = None
        print(f"[Train] Created MobileNetV3 backbone model")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def calculate_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> float:
    """Calculate mean Intersection over Union."""
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if ious else 0.0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Handle different model output formats
        if isinstance(outputs, dict):
            logits = outputs['out']
            loss = criterion(logits, masks)
            if 'aux' in outputs:
                loss += 0.4 * criterion(outputs['aux'], masks)
        else:
            logits = outputs
            loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_miou += calculate_miou(logits, masks)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss={loss.item():.4f}")
    
    return total_loss / len(dataloader), total_miou / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Handle different model output formats
            if isinstance(outputs, dict):
                logits = outputs['out']
            else:
                logits = outputs
            
            loss = criterion(logits, masks)
            total_loss += loss.item()
            total_miou += calculate_miou(logits, masks)
    
    return total_loss / len(dataloader), total_miou / len(dataloader)


def train(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = './checkpoints',
    resume: str = None,
    model_type: str = 'mobilenet'
):
    """Main training loop."""
    print("=" * 60)
    print("AeroSeg Training")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    
    # Create datasets
    train_dataset = SemanticDroneDataset(data_dir, split='train')
    val_dataset = SemanticDroneDataset(data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[Train] Training samples: {len(train_dataset)}")
    print(f"[Train] Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(num_classes=3, model_type=model_type, pretrained=(model_type == 'mobilenet'))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0
    
    if resume and Path(resume).exists():
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint.get('best_miou', 0.0)
        print(f"[Train] Resumed from epoch {start_epoch}")
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': []}
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_miou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_miou = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val mIoU:   {val_miou:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, checkpoint_path / 'best.pth')
            print(f"[✓] Saved best model (mIoU: {best_miou:.4f})")
        
        # Save latest
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
        }, checkpoint_path / 'latest.pth')
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print("=" * 60)
    
    # Plot training history
    plot_training_history(history, checkpoint_path / 'training_history.png')
    
    return model, history


def plot_training_history(history: Dict, save_path: Path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_miou'], label='Train')
    axes[1].plot(history['val_miou'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Mean IoU')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Saved training history plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train AeroSeg model')
    
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', '-c', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--model', '-m', type=str, default='mobilenet',
                       choices=['mobilenet', 'unet', 'light'],
                       help='Model architecture: mobilenet (pretrained), unet (custom), light (lightweight)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        model_type=args.model
    )


if __name__ == '__main__':
    main()
