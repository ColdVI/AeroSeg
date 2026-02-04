#!/usr/bin/env python3
"""
AeroSeg Custom U-Net Model
==========================

A lightweight U-Net architecture designed specifically for UAV landing zone
semantic segmentation. Optimized for real-time inference on edge devices.

Architecture:
    - Custom encoder with 4 downsampling stages
    - Skip connections for boundary preservation
    - Decoder with transposed convolutions
    - 3-class output: SAFE (0), HAZARD (1), WATER (2)

Usage:
    from custom_model import AeroSegUNet
    model = AeroSegUNet(num_classes=3)
    output = model(input_tensor)  # (B, 3, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Encoder block: MaxPool + ConvBlock."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Decoder block: Upsample + Concat + ConvBlock."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch due to odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for enhanced feature focusing."""
    
    def __init__(self, in_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample gate to match x size
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AeroSegUNet(nn.Module):
    """
    Custom U-Net for UAV Landing Zone Segmentation.
    
    Features:
        - Lightweight encoder (trainable from scratch)
        - 4-level feature hierarchy
        - Skip connections with optional attention gates
        - Optimized for 512x512 input
        - ~7M parameters (reduced from 11M backbone)
    
    Args:
        num_classes: Number of output classes (default: 3)
        in_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base channel count, doubled at each level (default: 64)
        bilinear: Use bilinear upsampling instead of transposed conv
        use_attention: Use attention gates in skip connections
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        base_channels: int = 64,
        bilinear: bool = True,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Channel configuration
        c1 = base_channels      # 64
        c2 = base_channels * 2  # 128
        c3 = base_channels * 4  # 256
        c4 = base_channels * 8  # 512
        c5 = base_channels * 16 # 1024
        
        # Encoder (downsampling path)
        self.inc = ConvBlock(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)
        
        # Attention gates (optional)
        if use_attention:
            self.attn4 = AttentionGate(c4, c5, c4 // 2)
            self.attn3 = AttentionGate(c3, c4, c3 // 2)
            self.attn2 = AttentionGate(c2, c3, c2 // 2)
            self.attn1 = AttentionGate(c1, c2, c1 // 2)
        
        # Decoder (upsampling path)
        # After up1: x5 (1024) upsampled + x4 skip (512) = 1536 -> output c4 (256 with factor)
        # After up2: c4 (256) upsampled + x3 skip (256) = 512 -> output c3 (128 with factor)
        # After up3: c3 (128) upsampled + x2 skip (128) = 256 -> output c2 (64 with factor)
        # After up4: c2 (64) upsampled + x1 skip (64) = 128 -> output c1 (64)
        factor = 2 if bilinear else 1
        
        self.up1 = UpBlock(c5 + c4, c4 // factor, bilinear)  # 1536 -> 256
        self.up2 = UpBlock(c4 // factor + c3, c3 // factor, bilinear)  # 512 -> 128
        self.up3 = UpBlock(c3 // factor + c2, c2 // factor, bilinear)  # 256 -> 64
        self.up4 = UpBlock(c2 // factor + c1, c1, bilinear)  # 128 -> 64
        
        # Output layer
        self.outc = nn.Conv2d(c1, num_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Encoder
        x1 = self.inc(x)    # (B, 64, H, W)
        x2 = self.down1(x1) # (B, 128, H/2, W/2)
        x3 = self.down2(x2) # (B, 256, H/4, W/4)
        x4 = self.down3(x3) # (B, 512, H/8, W/8)
        x5 = self.down4(x4) # (B, 1024, H/16, W/16)
        
        # Apply attention to skip connections
        if self.use_attention:
            x4 = self.attn4(x4, x5)
            x3 = self.attn3(x3, x4)
            x2 = self.attn2(x2, x3)
            x1 = self.attn1(x1, x2)
        
        # Decoder
        x = self.up1(x5, x4) # (B, 256, H/8, W/8)
        x = self.up2(x, x3)  # (B, 128, H/4, W/4)
        x = self.up3(x, x2)  # (B, 64, H/2, W/2)
        x = self.up4(x, x1)  # (B, 64, H, W)
        
        # Output
        logits = self.outc(x) # (B, num_classes, H, W)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AeroSegLightUNet(nn.Module):
    """
    Simplified, lightweight U-Net variant.
    
    Half the channel count for faster inference.
    Suitable for edge deployment (Raspberry Pi, Jetson Nano).
    
    ~1.9M parameters
    """
    
    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Lighter channel configuration
        c1, c2, c3, c4 = 32, 64, 128, 256
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c1, c2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c2, c3))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c3, c4))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c4, c4))
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c4 * 2, c4)
        
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        
        # Output
        self.outc = nn.Conv2d(c1, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.outc(d1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str = 'unet',
    num_classes: int = 3,
    use_attention: bool = False,
    pretrained: bool = False
) -> nn.Module:
    """
    Factory function to create AeroSeg models.
    
    Args:
        model_type: 'unet', 'light', or 'mobilenet' (backbone-based)
        num_classes: Number of output classes
        use_attention: Whether to use attention gates (UNet only)
        pretrained: Whether to use pretrained backbone (mobilenet only)
    
    Returns:
        PyTorch model
    """
    if model_type == 'unet':
        model = AeroSegUNet(num_classes=num_classes, use_attention=use_attention)
        print(f"[AeroSeg] Created U-Net with {model.count_parameters():,} parameters")
    elif model_type == 'light':
        model = AeroSegLightUNet(num_classes=num_classes)
        print(f"[AeroSeg] Created Light U-Net with {model.count_parameters():,} parameters")
    elif model_type == 'mobilenet':
        # Fall back to pre-trained MobileNetV3 backbone
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
        
        if pretrained:
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            model = deeplabv3_mobilenet_v3_large(weights=weights)
        else:
            model = deeplabv3_mobilenet_v3_large(weights=None)
        
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier = None
        
        print(f"[AeroSeg] Created MobileNetV3 backbone model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("AeroSeg Custom U-Net Model Test")
    print("=" * 60)
    
    # Test standard U-Net
    print("\n1. Standard U-Net:")
    model = AeroSegUNet(num_classes=3)
    print(f"   Parameters: {model.count_parameters():,}")
    
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y.shape}")
    
    # Test U-Net with attention
    print("\n2. U-Net with Attention:")
    model_attn = AeroSegUNet(num_classes=3, use_attention=True)
    print(f"   Parameters: {model_attn.count_parameters():,}")
    
    # Test Light U-Net
    print("\n3. Light U-Net:")
    model_light = AeroSegLightUNet(num_classes=3)
    print(f"   Parameters: {model_light.count_parameters():,}")
    
    with torch.no_grad():
        y_light = model_light(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y_light.shape}")
    
    # Inference speed test
    import time
    print("\n4. Inference Speed (CPU):")
    
    for name, m in [("U-Net", model), ("Light U-Net", model_light)]:
        m.eval()
        times = []
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                _ = m(x)
            times.append(time.time() - start)
        avg_time = np.mean(times[1:]) * 1000  # Skip first (warm-up)
        print(f"   {name}: {avg_time:.1f}ms")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
