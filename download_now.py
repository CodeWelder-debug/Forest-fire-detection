"""
Download fire and person datasets from Roboflow Universe.
Using verified public datasets.
"""

from roboflow import Roboflow
from pathlib import Path
import sys

def download_datasets():
    """Download datasets using Roboflow API."""
    print("=" * 60)
    print("DOWNLOADING DATASETS FROM ROBOFLOW")
    print("=" * 60)
    
    api_key = "mU68Yq2IDTUqkhuJfET6"
    base_dir = Path(__file__).parent
    raw_data_dir = base_dir / "raw_datasets"
    raw_data_dir.mkdir(exist_ok=True)
    
    rf = Roboflow(api_key=api_key)
    
    # Download fire dataset
    # Source: https://universe.roboflow.com/moh-wghvl/fire-detection-h0vjp/dataset/2
    print("\n📥 Downloading fire detection dataset...")
    print("   Workspace: moh-wghvl")
    print("   Project: fire-detection-h0vjp")
    print("   Version: 2")
    print("   Images: 612")
    
    try:
        project = rf.workspace("moh-wghvl").project("fire-detection-h0vjp")
        dataset = project.version(2).download("yolov8", location=str(raw_data_dir / "fire_dataset"))
        print("✅ Fire dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading fire dataset: {e}")
        print("\nTrying alternative fire dataset...")
        try:
            # Alternative: https://universe.roboflow.com/leilamegdiche/fire-detection-rsqrr
            project = rf.workspace("leilamegdiche").project("fire-detection-rsqrr")
            dataset = project.version(1).download("yolov8", location=str(raw_data_dir / "fire_dataset"))
            print("✅ Alternative fire dataset downloaded successfully!")
        except Exception as e2:
            print(f"❌ Error: {e2}")
            sys.exit(1)
    
    # Download person dataset
    # Source: https://universe.roboflow.com/digital-signal-processing/person-detection-dizkz/dataset/2
    print("\n📥 Downloading person detection dataset...")
    print("   Workspace: digital-signal-processing")
    print("   Project: person-detection-dizkz")
    print("   Version: 2")
    print("   Images: 2,212")
    
    try:
        project = rf.workspace("digital-signal-processing").project("person-detection-dizkz")
        dataset = project.version(2).download("yolov8", location=str(raw_data_dir / "person_dataset"))
        print("✅ Person dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading person dataset: {e}")
        print("\nTrying alternative person dataset...")
        try:
            # Alternative: https://universe.roboflow.com/abner/person-hgivm
            project = rf.workspace("abner").project("person-hgivm")
            dataset = project.version(1).download("yolov8", location=str(raw_data_dir / "person_dataset"))
            print("✅ Alternative person dataset downloaded successfully!")
        except Exception as e2:
            print(f"❌ Error: {e2}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ ALL DATASETS DOWNLOADED!")
    print("=" * 60)
    print(f"\nDatasets saved to: {raw_data_dir}")
    print(f"  Fire dataset: {raw_data_dir / 'fire_dataset'}")
    print(f"  Person dataset: {raw_data_dir / 'person_dataset'}")
    print("\nNext step: Run 'python prepare_yolo_dataset.py'")

if __name__ == "__main__":
    download_datasets()
