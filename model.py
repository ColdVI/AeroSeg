"""
AeroSeg Model Module
====================
DeepLabV3 with MobileNetV3 backbone for UAV landing zone segmentation.
Optimized for low-latency inference on edge devices.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from typing import Dict, Tuple, Optional
import numpy as np


class AeroSegModel:
    """
    Semantic segmentation model for aerial landing zone identification.
    
    Uses pre-trained DeepLabV3 with MobileNetV3-Large backbone (COCO dataset).
    Maps COCO classes to three UAV-relevant categories:
    - SAFE (0): Suitable for landing (grass, pavement, ground)
    - HAZARD (1): Obstacles to avoid (buildings, vehicles, people)
    - WATER (2): Water bodies (rivers, pools, sea)
    """
    
    # COCO class indices (21 classes including background)
    # Reference: https://pytorch.org/vision/stable/models.html#semantic-segmentation
    COCO_CLASSES = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor',
    }
    
    # Category mapping for UAV landing zone analysis
    CATEGORY_SAFE = 0      # Green - safe to land
    CATEGORY_HAZARD = 1    # Red - obstacle/danger
    CATEGORY_WATER = 2     # Blue - water body
    
    # COCO classes mapped to HAZARD (obstacles)
    HAZARD_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    
    # COCO classes mapped to WATER (currently no direct water class in VOC/COCO)
    # Water detection would require additional training - for now, background near blue is approximated
    WATER_CLASSES = set()  # Placeholder for future fine-tuning
    
    def __init__(self, device: Optional[str] = None, checkpoint_path: Optional[str] = None, model_type: str = 'mobilenet'):
        """
        Initialize the AeroSeg model.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', or 'mps').
                   Auto-detects if not specified.
            checkpoint_path: Path to fine-tuned checkpoint (.pth file).
                            If None, uses pre-trained COCO weights.
            model_type: 'mobilenet', 'unet', or 'light'.
        """
        self.device = self._get_device(device)
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.target_size = (512, 512)  # Standard size for custom U-Net to avoid OOM
        self.model = self._load_model()
        self.model.eval()
        
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Auto-detect the best available device."""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def _load_model(self) -> nn.Module:
        """Load segmentation model."""
        if self.model_type == 'mobilenet':
            print(f"[AeroSeg] Loading DeepLabV3-MobileNetV3 on {self.device}...")
            if self.checkpoint_path:
                print(f"[AeroSeg] Loading fine-tuned checkpoint: {self.checkpoint_path}")
                model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
                model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)
                model.aux_classifier = None
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights, progress=True)
        
        elif self.model_type in ['unet', 'light']:
            from custom_model import create_model
            print(f"[AeroSeg] Loading {self.model_type} model on {self.device}...")
            model = create_model(model_type=self.model_type, num_classes=3)
            
            if self.checkpoint_path:
                print(f"[AeroSeg] Loading fine-tuned checkpoint: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("[WARNING] Custom models should be used with a checkpoint for meaningful results.")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model.to(self.device)

    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: RGB image as numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Resize to target size for custom models or if image is very large
        import cv2
        img_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Convert to float and normalize
        img = img_resized.astype(np.float32) / 255.0
        img = (img - mean) / std
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Add batch dimension
        return img_tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform semantic segmentation on an aerial image.
        
        Args:
            image: RGB image as numpy array (H, W, C)
            
        Returns:
            Tuple of:
            - raw_mask: Raw COCO class predictions (H, W)
            - category_mask: Mapped to SAFE/HAZARD/WATER categories (H, W)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        output = self.model(input_tensor)
        
        if isinstance(output, dict):
            output = output['out']
        
        # Get class predictions
        raw_mask_small = output.argmax(dim=1).squeeze().cpu().numpy()
        
        # Resize mask back to original image size
        import cv2
        raw_mask = cv2.resize(
            raw_mask_small, 
            (image.shape[1], image.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Map to UAV categories
        category_mask = self._map_to_categories(raw_mask)
        
        return raw_mask, category_mask
    
    def _map_to_categories(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Map COCO class predictions to UAV landing categories.
        
        Args:
            raw_mask: Raw COCO class predictions (H, W)
            
        Returns:
            Category mask (H, W) with values 0=SAFE, 1=HAZARD, 2=WATER
        """
        category_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        
        # Default is SAFE (background and ground-like classes)
        # Mark HAZARD classes
        for class_id in self.HAZARD_CLASSES:
            category_mask[raw_mask == class_id] = self.CATEGORY_HAZARD
            
        # Mark WATER classes (if any)
        for class_id in self.WATER_CLASSES:
            category_mask[raw_mask == class_id] = self.CATEGORY_WATER
            
        return category_mask
    
    def get_category_stats(self, category_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate pixel statistics for each category.
        
        Args:
            category_mask: Category mask (H, W)
            
        Returns:
            Dictionary with percentage of pixels in each category
        """
        total_pixels = category_mask.size
        
        safe_pixels = np.sum(category_mask == self.CATEGORY_SAFE)
        hazard_pixels = np.sum(category_mask == self.CATEGORY_HAZARD)
        water_pixels = np.sum(category_mask == self.CATEGORY_WATER)
        
        return {
            'safe_percent': (safe_pixels / total_pixels) * 100,
            'hazard_percent': (hazard_pixels / total_pixels) * 100,
            'water_percent': (water_pixels / total_pixels) * 100,
            'safe_pixels': int(safe_pixels),
            'hazard_pixels': int(hazard_pixels),
            'water_pixels': int(water_pixels),
            'total_pixels': int(total_pixels)
        }
