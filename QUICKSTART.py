"""
Quick Start Guide - Fire and Person Detection System
Follow these steps to get started with the detection system.
"""

import os
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

def main():
    """Display quick start guide."""
    print_header("FIRE AND PERSON DETECTION - QUICK START GUIDE")
    
    print("\n📋 OVERVIEW")
    print("-" * 70)
    print("This system detects fire and person in real-time using YOLOv8.")
    print("All necessary scripts have been created in this directory.")
    
    print("\n📦 STEP 1: GET ROBOFLOW API KEY (FREE)")
    print("-" * 70)
    print("1. Visit: https://app.roboflow.com/")
    print("2. Sign up for a FREE account (takes 1 minute)")
    print("3. Click on your profile → Settings → Roboflow API")
    print("4. Copy your API key")
    
    print("\n🔥 STEP 2: DOWNLOAD DATASETS")
    print("-" * 70)
    print("Run: python download_datasets.py")
    print("  - Enter your Roboflow API key when prompted")
    print("  - Fire and person datasets will be downloaded automatically")
    print("  - Datasets are public and free to use")
    
    print("\n🔧 STEP 3: PREPARE DATASET")
    print("-" * 70)
    print("Run: python prepare_yolo_dataset.py")
    print("  - Combines fire and person datasets")
    print("  - Creates train/validation splits (80/20)")
    print("  - Generates YOLO configuration file")
    
    print("\n🚀 STEP 4: TRAIN MODEL")
    print("-" * 70)
    print("Run: python train_model.py")
    print("  - Choose model size (n=nano is fastest, recommended for testing)")
    print("  - Set epochs (50 is default, 10-20 for quick testing)")
    print("  - Training time depends on hardware (GPU recommended)")
    print("  - Results saved to: runs/detect/fire_person_detection/")
    
    print("\n📹 STEP 5: RUN WEBCAM DETECTION")
    print("-" * 70)
    print("Run: python webcam_inference.py")
    print("  - Enter camera index (1 as specified)")
    print("  - Real-time detection with bounding boxes")
    print("  - Press 'q' to quit, 's' to save screenshot")
    
    print("\n⚡ QUICK START (AUTOMATED)")
    print("-" * 70)
    print("Run: python run_pipeline.py")
    print("  - Runs all steps automatically")
    print("  - You'll need your Roboflow API key")
    
    print("\n📚 ALTERNATIVE: MANUAL DATASET DOWNLOAD")
    print("-" * 70)
    print("If you prefer to download datasets manually:")
    print("\n1. Fire Dataset:")
    print("   URL: https://universe.roboflow.com/fire-detection/fire-detection-dataset")
    print("   Format: YOLOv8")
    print("   Extract to: raw_datasets/fire_dataset/")
    print("\n2. Person Dataset:")
    print("   URL: https://universe.roboflow.com/coco-dataset/coco-person")
    print("   Format: YOLOv8")
    print("   Extract to: raw_datasets/person_dataset/")
    
    print("\n💡 TIPS")
    print("-" * 70)
    print("• Use GPU for faster training (CUDA will be auto-detected)")
    print("• Start with small model (nano) and few epochs for testing")
    print("• Check runs/detect/fire_person_detection/ for training results")
    print("• Model weights saved as best.pt and last.pt")
    
    print("\n🔍 TROUBLESHOOTING")
    print("-" * 70)
    print("• Camera not found: Try different camera indices (0, 1, 2)")
    print("• Out of memory: Reduce batch size or use smaller model")
    print("• Slow training: Reduce epochs or image size")
    
    print("\n📁 PROJECT STRUCTURE")
    print("-" * 70)
    base_dir = Path(__file__).parent
    
    files = {
        "download_datasets.py": "Download fire and person datasets",
        "prepare_yolo_dataset.py": "Prepare combined YOLO dataset",
        "train_model.py": "Train YOLOv8 model",
        "webcam_inference.py": "Real-time webcam detection",
        "run_pipeline.py": "Automated pipeline (all steps)",
        "requirements.txt": "Python dependencies",
        "README.md": "Detailed documentation"
    }
    
    for file, desc in files.items():
        status = "✅" if (base_dir / file).exists() else "❌"
        print(f"{status} {file:30s} - {desc}")
    
    print("\n" + "=" * 70)
    print("READY TO START!".center(70))
    print("=" * 70)
    print("\nNext step: Get your Roboflow API key and run download_datasets.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
