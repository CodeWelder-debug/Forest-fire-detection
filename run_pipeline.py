"""
Automated pipeline to download datasets, prepare them, and train the model.
This script runs the entire pipeline from start to finish.
"""

import os
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

def download_datasets_roboflow():
    """Download datasets using Roboflow API."""
    print_header("STEP 1: DOWNLOADING DATASETS")
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\nERROR: Roboflow library not found.")
        print("Installing roboflow...")
        os.system("pip install roboflow")
        from roboflow import Roboflow
    
    print("\nTo download datasets, you need a free Roboflow API key.")
    print("1. Visit: https://app.roboflow.com/")
    print("2. Sign up for a free account")
    print("3. Go to Settings and copy your API key")
    
    api_key = input("\nEnter your Roboflow API key: ").strip()
    
    if not api_key:
        print("\nNo API key provided. Cannot proceed with automatic download.")
        print("Please run 'python download_datasets.py' manually.")
        return False
    
    base_dir = Path(__file__).parent
    raw_data_dir = base_dir / "raw_datasets"
    raw_data_dir.mkdir(exist_ok=True)
    
    rf = Roboflow(api_key=api_key)
    
    # Download fire dataset
    print("\n📥 Downloading fire detection dataset...")
    try:
        project = rf.workspace("roboflow-universe-projects").project("fire-detection-nnvxp")
        dataset = project.version(1).download("yolov8", location=str(raw_data_dir / "fire_dataset"))
        print("✅ Fire dataset downloaded!")
    except Exception as e:
        print(f"❌ Error downloading fire dataset: {e}")
        print("\nTrying alternative fire dataset...")
        try:
            project = rf.workspace("fire-smoke-detection").project("fire-and-smoke-detection")
            dataset = project.version(1).download("yolov8", location=str(raw_data_dir / "fire_dataset"))
            print("✅ Alternative fire dataset downloaded!")
        except Exception as e2:
            print(f"❌ Error: {e2}")
            return False
    
    # Download person dataset
    print("\n📥 Downloading person detection dataset...")
    try:
        project = rf.workspace("roboflow-universe-projects").project("person-detection-o4rdr")
        dataset = project.version(2).download("yolov8", location=str(raw_data_dir / "person_dataset"))
        print("✅ Person dataset downloaded!")
    except Exception as e:
        print(f"❌ Error downloading person dataset: {e}")
        print("\nTrying alternative person dataset...")
        try:
            project = rf.workspace("coco-dataset").project("coco-person")
            dataset = project.version(1).download("yolov8", location=str(raw_data_dir / "person_dataset"))
            print("✅ Alternative person dataset downloaded!")
        except Exception as e2:
            print(f"❌ Error: {e2}")
            return False
    
    print("\n✅ All datasets downloaded successfully!")
    return True

def prepare_dataset():
    """Prepare the combined dataset."""
    print_header("STEP 2: PREPARING DATASET")
    
    print("\nRunning dataset preparation script...")
    result = os.system("python prepare_yolo_dataset.py")
    
    if result == 0:
        print("\n✅ Dataset prepared successfully!")
        return True
    else:
        print("\n❌ Error preparing dataset.")
        return False

def train_model():
    """Train the YOLOv8 model."""
    print_header("STEP 3: TRAINING MODEL")
    
    print("\nStarting model training...")
    print("This may take a while depending on your hardware.")
    print("You can monitor progress in the terminal output.")
    
    result = os.system("python train_model.py")
    
    if result == 0:
        print("\n✅ Model trained successfully!")
        return True
    else:
        print("\n❌ Error training model.")
        return False

def run_inference():
    """Run webcam inference."""
    print_header("STEP 4: RUNNING WEBCAM INFERENCE")
    
    print("\nStarting webcam inference...")
    print("Press 'q' to quit the webcam window.")
    
    os.system("python webcam_inference.py")

def main():
    """Main pipeline function."""
    print_header("FIRE AND PERSON DETECTION PIPELINE")
    
    print("\nThis script will:")
    print("1. Download fire and person datasets from Roboflow")
    print("2. Prepare and combine datasets in YOLO format")
    print("3. Train YOLOv8 model")
    print("4. Run real-time webcam inference")
    
    choice = input("\nDo you want to proceed? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("\nPipeline cancelled.")
        return
    
    # Step 1: Download datasets
    if not download_datasets_roboflow():
        print("\n❌ Pipeline failed at dataset download step.")
        print("You can manually download datasets and run individual scripts.")
        return
    
    # Step 2: Prepare dataset
    if not prepare_dataset():
        print("\n❌ Pipeline failed at dataset preparation step.")
        return
    
    # Step 3: Train model
    train_choice = input("\nDo you want to train the model now? (y/n): ").strip().lower()
    
    if train_choice == 'y':
        if not train_model():
            print("\n❌ Pipeline failed at model training step.")
            return
        
        # Step 4: Run inference
        inference_choice = input("\nDo you want to run webcam inference now? (y/n): ").strip().lower()
        
        if inference_choice == 'y':
            run_inference()
    
    print_header("PIPELINE COMPLETE")
    
    print("\n✅ All steps completed successfully!")
    print("\nYou can now:")
    print("  - Run webcam inference: python webcam_inference.py")
    print("  - Retrain model: python train_model.py")
    print("  - Check results: runs/detect/fire_person_detection/")

if __name__ == "__main__":
    main()
