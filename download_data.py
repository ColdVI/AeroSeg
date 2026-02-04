#!/usr/bin/env python3
"""
Dataset Download & Setup Script for AeroSeg
============================================

Downloads and prepares aerial segmentation datasets for training.

Usage:
    python download_data.py --dataset semantic-drone
    python download_data.py --dataset uavid
    python download_data.py --list
"""

import os
import argparse
import shutil
from pathlib import Path


# Dataset information
DATASETS = {
    "semantic-drone": {
        "name": "Semantic Drone Dataset",
        "description": "Urban aerial imagery from Graz University (400 train, 200 test images)",
        "kaggle_id": "bulentsiyah/semantic-drone-dataset",
        "alternative_kaggle": "santurini/semantic-drone-dataset",
        "homepage": "http://dronedataset.icg.tugraz.at",
        "classes": 20,
        "resolution": "6000x4000 (24 Mpx)",
        "altitude": "5-30 meters"
    },
    "semantic-drone-lite": {
        "name": "Semantic Drone Dataset (Lite)",
        "description": "Preprocessed version with 5 macro-classes, 960x736px resolution",
        "kaggle_id": "amanullahasraf/semantic-segmentation-drone-dataset",
        "classes": 5,
        "resolution": "960x736"
    },
    "uavid": {
        "name": "UAVid Dataset",
        "description": "Urban scene understanding for UAVs (8 classes)",
        "homepage": "https://uavid.nl/",
        "classes": 8,
        "note": "Requires registration on official website"
    },
    "aeroscapes": {
        "name": "AeroScapes Dataset",
        "description": "Aerial scene parsing dataset",
        "github": "https://github.com/ishann/aeroscapes",
        "classes": 11
    }
}


def list_datasets():
    """List all available datasets with details."""
    print("\n" + "=" * 70)
    print("AVAILABLE AERIAL SEGMENTATION DATASETS")
    print("=" * 70 + "\n")
    
    for key, info in DATASETS.items():
        print(f"📦 {info['name']}")
        print(f"   ID: {key}")
        print(f"   Description: {info['description']}")
        print(f"   Classes: {info['classes']}")
        
        if 'kaggle_id' in info:
            print(f"   Kaggle: kaggle datasets download -d {info['kaggle_id']}")
        if 'homepage' in info:
            print(f"   Homepage: {info['homepage']}")
        if 'github' in info:
            print(f"   GitHub: {info['github']}")
        if 'note' in info:
            print(f"   ⚠️  Note: {info['note']}")
        print()
    
    print("=" * 70)
    print("\nTo download, use: python download_data.py --dataset <dataset-id>")
    print("Requires: pip install kaggle (and ~/.kaggle/kaggle.json API key)")
    print("=" * 70 + "\n")


def setup_data_directory(base_path: Path):
    """Create the data directory structure."""
    dirs = [
        base_path / "train" / "images",
        base_path / "train" / "masks",
        base_path / "val" / "images",
        base_path / "val" / "masks",
        base_path / "test" / "images",
        base_path / "test" / "masks",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[✓] Created: {d}")
    
    return base_path


def download_kaggle_dataset(kaggle_id: str, output_path: Path):
    """Download dataset from Kaggle."""
    try:
        import kaggle
        print(f"[↓] Downloading from Kaggle: {kaggle_id}")
        kaggle.api.dataset_download_files(kaggle_id, path=output_path, unzip=True)
        print(f"[✓] Downloaded to: {output_path}")
        return True
    except ImportError:
        print("[!] Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        print("\nManual download instructions:")
        print(f"  1. Visit: https://www.kaggle.com/datasets/{kaggle_id}")
        print(f"  2. Download and extract to: {output_path}")
        return False


def download_semantic_drone(output_path: Path):
    """Download and organize Semantic Drone Dataset."""
    dataset = DATASETS["semantic-drone"]
    
    print(f"\n{'=' * 60}")
    print(f"DOWNLOADING: {dataset['name']}")
    print(f"{'=' * 60}")
    print(f"Resolution: {dataset['resolution']}")
    print(f"Classes: {dataset['classes']}")
    print(f"Altitude: {dataset['altitude']}")
    print(f"{'=' * 60}\n")
    
    # Try primary Kaggle ID
    raw_path = output_path / "raw"
    raw_path.mkdir(parents=True, exist_ok=True)
    
    success = download_kaggle_dataset(dataset['kaggle_id'], raw_path)
    
    if not success:
        # Try alternative
        print(f"\n[↓] Trying alternative source...")
        success = download_kaggle_dataset(dataset['alternative_kaggle'], raw_path)
    
    if success:
        print("\n[✓] Download complete!")
        print(f"\nNext steps:")
        print(f"  1. Organize data into train/val/test splits")
        print(f"  2. Run: python train.py --data {output_path}")
    else:
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 60)
        print(f"\n1. Visit: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset")
        print(f"   OR: {dataset['homepage']}")
        print(f"\n2. Download and extract to: {raw_path}")
        print(f"\n3. Expected structure after extraction:")
        print(f"   {raw_path}/")
        print(f"   ├── dataset/")
        print(f"   │   └── semantic_drone_dataset/")
        print(f"   │       ├── original_images/")
        print(f"   │       └── label_images_semantic/")
        print("=" * 60)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download aerial segmentation datasets for AeroSeg training"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=list(DATASETS.keys()),
        help="Dataset to download"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    if args.list or not args.dataset:
        list_datasets()
        return
    
    output_path = Path(args.output).resolve()
    print(f"\n[i] Output directory: {output_path}")
    
    # Setup directory structure
    setup_data_directory(output_path)
    
    # Download selected dataset
    if args.dataset == "semantic-drone":
        download_semantic_drone(output_path)
    elif args.dataset == "semantic-drone-lite":
        download_kaggle_dataset(
            DATASETS["semantic-drone-lite"]["kaggle_id"],
            output_path / "raw"
        )
    elif args.dataset == "uavid":
        print(f"\n⚠️  UAVid requires manual registration at: https://uavid.nl/")
        print(f"Download and extract to: {output_path}/raw")
    elif args.dataset == "aeroscapes":
        print(f"\n⚠️  AeroScapes is available at: {DATASETS['aeroscapes']['github']}")
        print(f"Clone and extract to: {output_path}/raw")


if __name__ == "__main__":
    main()
