#!/usr/bin/env python3
"""
AeroSeg: Autonomous UAV Landing Zone & Obstacle Identification System
======================================================================

CLI tool for processing aerial images/videos to identify safe landing zones.

Usage:
    python main.py --image <path>           # Process single image
    python main.py --video <path>           # Process video file
    python main.py --image <path> --output <path>  # Save result
    
Example:
    python main.py --image aerial_view.jpg --output result.png
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import AeroSegModel
from processor import ImageProcessor, SafetyResult


class AeroSegPipeline:
    """
    Main pipeline for UAV landing zone analysis.
    
    Combines the AeroSegModel and ImageProcessor for end-to-end
    image/video processing with safety assessment.
    """
    
    def __init__(self, roi_size: int = 200, device: str = None, checkpoint_path: str = None, model_type: str = 'mobilenet'):
        """
        Initialize the AeroSeg pipeline.
        
        Args:
            roi_size: Size of the central ROI for safety analysis
            device: Device for model inference ('cuda', 'cpu', 'mps')
            checkpoint_path: Path to fine-tuned checkpoint (uses COCO if None)
            model_type: Architecture type ('mobilenet', 'unet', 'light')
        """
        print("=" * 60)
        print("AeroSeg: Autonomous UAV Landing Zone Identification")
        print("=" * 60)
        
        self.model = AeroSegModel(device=device, checkpoint_path=checkpoint_path, model_type=model_type)
        self.processor = ImageProcessor(roi_size=roi_size)
        
        print(f"[AeroSeg] ROI Size: {roi_size}x{roi_size} pixels")
        print(f"[AeroSeg] Safety Threshold: {self.processor.safety_threshold}%")
        print("=" * 60 + "\n")
        
    def process_image(
        self, 
        image_path: str, 
        output_path: str = None,
        show: bool = True
    ) -> SafetyResult:
        """
        Process a single aerial image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            show: Whether to display the result
            
        Returns:
            SafetyResult with analysis
        """
        # Load image
        print(f"[AeroSeg] Loading image: {image_path}")
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        print("[AeroSeg] Running semantic segmentation...")
        start_time = time.time()
        
        raw_mask, category_mask = self.model.segment(image_rgb)
        
        inference_time = (time.time() - start_time) * 1000
        print(f"[AeroSeg] Inference completed in {inference_time:.2f}ms")
        
        # Process and visualize
        visualization, safety_result = self.processor.process_frame(
            image_rgb, category_mask
        )
        
        # Print results
        self._print_results(safety_result, inference_time)
        
        # Save if output path specified
        if output_path:
            # Convert RGB to BGR for OpenCV
            output_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), output_bgr)
            print(f"[AeroSeg] Saved result to: {output_path}")
        
        # Display
        if show:
            self._display_result(image_rgb, visualization, safety_result)
            
        return safety_result
    
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        show: bool = True
    ) -> None:
        """
        Process a video file frame by frame.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save processed video
            show: Whether to display frames in real-time
        """
        print(f"[AeroSeg] Loading video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[AeroSeg] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run segmentation
                _, category_mask = self.model.segment(frame_rgb)
                
                # Process frame
                visualization, safety_result = self.processor.process_frame(
                    frame_rgb, category_mask
                )
                
                # Convert back to BGR for display/save
                vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                
                # Add frame info
                cv2.putText(
                    vis_bgr,
                    f"Frame: {frame_count}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Write to output
                if writer:
                    writer.write(vis_bgr)
                
                # Display
                if show:
                    cv2.imshow('AeroSeg - Landing Zone Analysis', vis_bgr)
                    
                    # Print status periodically
                    if frame_count % 30 == 0:
                        print(f"\r[Frame {frame_count}/{total_frames}] {safety_result.status_message}", end='')
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[AeroSeg] Video processing interrupted by user")
                        break
                        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
        print(f"\n[AeroSeg] Processed {frame_count} frames")
        if output_path:
            print(f"[AeroSeg] Saved video to: {output_path}")
    
    def _print_results(self, result: SafetyResult, inference_time: float) -> None:
        """Print safety analysis results to console."""
        print("\n" + "=" * 60)
        print("LANDING ZONE ANALYSIS RESULTS")
        print("=" * 60)
        print(f"  Safe Area:    {result.safe_percent:6.2f}%")
        print(f"  Hazard Area:  {result.hazard_percent:6.2f}%")
        print(f"  Water Area:   {result.water_percent:6.2f}%")
        print(f"  Safety Score: {result.safety_score:6.2f}")
        print("-" * 60)
        print(f"  Inference:    {inference_time:.2f}ms")
        print("=" * 60)
        print(f"\n  >>> {result.status_message} <<<\n")
        print("=" * 60)
    
    def _display_result(
        self,
        original: np.ndarray,
        visualization: np.ndarray,
        result: SafetyResult
    ) -> None:
        """Display the result using matplotlib."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title('Original Aerial View', fontsize=12)
        axes[0].axis('off')
        
        # Segmentation overlay
        axes[1].imshow(visualization)
        status_color = 'green' if result.is_safe else 'red'
        axes[1].set_title(
            f'Landing Zone Analysis - {result.status_message}',
            fontsize=12,
            color=status_color
        )
        axes[1].axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.5, label='Safe'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.5, label='Hazard'),
            plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.5, label='Water'),
        ]
        axes[1].legend(
            handles=legend_elements,
            loc='lower right',
            fontsize=10
        )
        
        plt.suptitle(
            'AeroSeg: UAV Landing Zone Identification System',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='AeroSeg: Autonomous UAV Landing Zone Identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image aerial.jpg
  python main.py --image aerial.jpg --output result.png
  python main.py --video flight.mp4 --output processed.mp4
  python main.py --image aerial.jpg --roi-size 300 --device cuda
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to input image file'
    )
    input_group.add_argument(
        '--video', '-v',
        type=str,
        help='Path to input video file'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save output file'
    )
    
    # Processing options
    parser.add_argument(
        '--roi-size',
        type=int,
        default=200,
        help='Size of central ROI for safety analysis (default: 200)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps'],
        default=None,
        help='Device for inference (default: auto-detect)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable visualization display'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to fine-tuned checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='mobilenet',
        choices=['mobilenet', 'unet', 'light'],
        help='Model architecture to use (mobilenet, unet, light)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AeroSegPipeline(
        roi_size=args.roi_size,
        device=args.device,
        checkpoint_path=args.checkpoint,
        model_type=args.model
    )
    
    # Process based on input type
    try:
        if args.image:
            result = pipeline.process_image(
                args.image,
                output_path=args.output,
                show=not args.no_display
            )
        else:
            pipeline.process_video(
                args.video,
                output_path=args.output,
                show=not args.no_display
            )
            
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        raise


if __name__ == '__main__':
    main()
