"""
AeroSeg Processor Module
========================
Image processing, ROI analysis, and visualization for UAV landing zone assessment.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SafetyResult:
    """Result of safety analysis for a region."""
    is_safe: bool
    safety_score: float
    safe_percent: float
    hazard_percent: float
    water_percent: float
    status_message: str
    
    
class ImageProcessor:
    """
    Processes segmentation results for UAV landing zone analysis.
    
    Handles:
    - Central ROI extraction for landing zone focus
    - Safety score calculation
    - Visualization overlay generation
    """
    
    # Category color mapping (BGR for OpenCV)
    COLORS = {
        0: (0, 255, 0),    # SAFE - Green
        1: (0, 0, 255),    # HAZARD - Red
        2: (255, 0, 0),    # WATER - Blue
    }
    
    # Safety threshold (percentage of safe pixels required)
    SAFETY_THRESHOLD = 70.0  # 70% safe pixels required for SECURE status
    
    def __init__(self, roi_size: int = 200, safety_threshold: float = None):
        """
        Initialize the image processor.
        
        Args:
            roi_size: Size of the central ROI square (default 200x200)
            safety_threshold: Minimum safe pixel percentage for SECURE status
        """
        self.roi_size = roi_size
        self.safety_threshold = safety_threshold or self.SAFETY_THRESHOLD
        
    def extract_central_roi(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the central region of interest from image and mask.
        
        This simulates a UAV's downward camera focusing on the landing zone
        directly below the aircraft.
        
        Args:
            image: Original image (H, W, C)
            mask: Segmentation mask (H, W)
            
        Returns:
            Tuple of (roi_image, roi_mask)
        """
        h, w = image.shape[:2]
        
        # Calculate center crop coordinates
        center_y, center_x = h // 2, w // 2
        half_size = self.roi_size // 2
        
        # Ensure ROI fits within image bounds
        y1 = max(0, center_y - half_size)
        y2 = min(h, center_y + half_size)
        x1 = max(0, center_x - half_size)
        x2 = min(w, center_x + half_size)
        
        roi_image = image[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]
        
        return roi_image, roi_mask
    
    def calculate_safety_score(self, category_mask: np.ndarray) -> SafetyResult:
        """
        Calculate the safety score for a region.
        
        The safety score is based on the percentage of pixels classified as SAFE
        within the central ROI.
        
        Args:
            category_mask: Category mask for the ROI (H, W)
            
        Returns:
            SafetyResult containing safety metrics and status
        """
        total_pixels = category_mask.size
        
        if total_pixels == 0:
            return SafetyResult(
                is_safe=False,
                safety_score=0.0,
                safe_percent=0.0,
                hazard_percent=0.0,
                water_percent=0.0,
                status_message="[SAFETY STATUS: ERROR - No pixels in ROI]"
            )
        
        # Count pixels by category
        safe_pixels = np.sum(category_mask == 0)
        hazard_pixels = np.sum(category_mask == 1)
        water_pixels = np.sum(category_mask == 2)
        
        # Calculate percentages
        safe_percent = (safe_pixels / total_pixels) * 100
        hazard_percent = (hazard_pixels / total_pixels) * 100
        water_percent = (water_pixels / total_pixels) * 100
        
        # Safety score is primarily safe percentage, penalized by hazards
        # Water is considered moderately unsafe
        safety_score = safe_percent - (hazard_percent * 0.5) - (water_percent * 0.3)
        safety_score = max(0.0, min(100.0, safety_score))
        
        # Determine status
        is_safe = safe_percent >= self.safety_threshold and hazard_percent < 20
        
        if is_safe:
            status_message = "[SAFETY STATUS: SECURE]"
        else:
            status_message = "[SAFETY STATUS: HAZARD DETECTED]"
            
        return SafetyResult(
            is_safe=is_safe,
            safety_score=safety_score,
            safe_percent=safe_percent,
            hazard_percent=hazard_percent,
            water_percent=water_percent,
            status_message=status_message
        )
    
    def create_overlay(
        self, 
        image: np.ndarray, 
        category_mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create a visualization overlay showing safe/hazard zones.
        
        Args:
            image: Original RGB image (H, W, C)
            category_mask: Category mask (H, W)
            alpha: Transparency of the overlay (0-1)
            
        Returns:
            Overlay image with colored regions (H, W, C)
        """
        # Ensure mask matches image dimensions
        if category_mask.shape[:2] != image.shape[:2]:
            category_mask = cv2.resize(
                category_mask, 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Create colored overlay
        overlay = np.zeros_like(image)
        
        for category, color in self.COLORS.items():
            # Convert BGR to RGB for overlay
            rgb_color = (color[2], color[1], color[0])
            overlay[category_mask == category] = rgb_color
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def draw_roi_box(
        self, 
        image: np.ndarray, 
        safety_result: SafetyResult
    ) -> np.ndarray:
        """
        Draw the ROI bounding box and safety status on the image.
        
        Args:
            image: Image to draw on (H, W, C)
            safety_result: Safety analysis result
            
        Returns:
            Image with ROI box and status overlay
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # Calculate ROI box coordinates
        center_y, center_x = h // 2, w // 2
        half_size = self.roi_size // 2
        
        y1 = max(0, center_y - half_size)
        y2 = min(h, center_y + half_size)
        x1 = max(0, center_x - half_size)
        x2 = min(w, center_x + half_size)
        
        # Draw ROI box
        color = (0, 255, 0) if safety_result.is_safe else (0, 0, 255)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # Draw status text
        status_text = "SECURE" if safety_result.is_safe else "HAZARD"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Background for text
        text_size = cv2.getTextSize(status_text, font, font_scale, thickness)[0]
        text_x = x1
        text_y = y1 - 10
        
        cv2.rectangle(
            result, 
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color, 
            -1
        )
        cv2.putText(
            result, 
            status_text, 
            (text_x, text_y),
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness
        )
        
        # Draw safety score
        score_text = f"Safety: {safety_result.safety_score:.1f}%"
        cv2.putText(
            result,
            score_text,
            (x1, y2 + 30),
            font,
            0.7,
            color,
            2
        )
        
        return result
    
    def process_frame(
        self,
        image: np.ndarray,
        category_mask: np.ndarray
    ) -> Tuple[np.ndarray, SafetyResult]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            image: Original RGB image (H, W, C)
            category_mask: Category mask from model (H, W)
            
        Returns:
            Tuple of (visualization, safety_result)
        """
        # Extract central ROI
        _, roi_mask = self.extract_central_roi(image, category_mask)
        
        # Calculate safety score for ROI
        safety_result = self.calculate_safety_score(roi_mask)
        
        # Create overlay visualization
        overlay = self.create_overlay(image, category_mask)
        
        # Draw ROI box and status
        visualization = self.draw_roi_box(overlay, safety_result)
        
        return visualization, safety_result
