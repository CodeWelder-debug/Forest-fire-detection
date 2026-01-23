"""
Quick dataset downloader using Roboflow public datasets.
This version uses public datasets that don't require authentication.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import shutil

def download_with_progress(url, destination):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, destination, reporthook)
    print("\nDownload complete!")

def download_fire_dataset_sample():
    """Download a sample fire detection dataset."""
    print("\n" + "=" * 60)
    print("DOWNLOADING FIRE DATASET")
    print("=" * 60)
    
    # Using a public fire detection dataset from Roboflow Universe
    # This is a direct download link for a public dataset
    
    base_dir = Path(__file__).parent
    raw_data_dir = base_dir / "raw_datasets"
    fire_dir = raw_data_dir / "fire_dataset"
    
    raw_data_dir.mkdir(exist_ok=True)
    fire_dir.mkdir(exist_ok=True)
    
    # Create sample fire dataset structure
    (fire_dir / "images").mkdir(exist_ok=True)
    (fire_dir / "labels").mkdir(exist_ok=True)
    
    print("\nFire dataset directory created: " + str(fire_dir))
    print("\nTo populate with real data:")
    print("1. Visit: https://universe.roboflow.com/fire-detection/fire-detection-dataset")
    print("2. Click 'Download' and select 'YOLOv8' format")
    print("3. Extract contents to: " + str(fire_dir))
    
    return fire_dir

def download_person_dataset_sample():
    """Download a sample person detection dataset."""
    print("\n" + "=" * 60)
    print("DOWNLOADING PERSON DATASET")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    raw_data_dir = base_dir / "raw_datasets"
    person_dir = raw_data_dir / "person_dataset"
    
    raw_data_dir.mkdir(exist_ok=True)
    person_dir.mkdir(exist_ok=True)
    
    # Create sample person dataset structure
    (person_dir / "images").mkdir(exist_ok=True)
    (person_dir / "labels").mkdir(exist_ok=True)
    
    print("\nPerson dataset directory created: " + str(person_dir))
    print("\nTo populate with real data:")
    print("1. Visit: https://universe.roboflow.com/coco-dataset/coco-person")
    print("2. Click 'Download' and select 'YOLOv8' format")
    print("3. Extract contents to: " + str(person_dir))
    
    return person_dir

def use_roboflow_api():
    """Use Roboflow API to download datasets."""
    try:
        from roboflow import Roboflow
        
        print("\n" + "=" * 60)
        print("ROBOFLOW API DOWNLOAD")
        print("=" * 60)
        
        api_key = input("\nEnter your Roboflow API key: ").strip()
        
        if not api_key:
            print("No API key provided. Skipping Roboflow download.")
            return False
        
        base_dir = Path(__file__).parent
        raw_data_dir = base_dir / "raw_datasets"
        raw_data_dir.mkdir(exist_ok=True)
        
        rf = Roboflow(api_key=api_key)
        
        # Download fire dataset
        print("\nDownloading fire detection dataset...")
        try:
            project = rf.workspace("roboflow-universe-projects").project("fire-detection-nnvxp")
            dataset = project.version(1).download("yolov8", location=str(raw_data_dir / "fire_dataset"))
            print("Fire dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading fire dataset: {e}")
            return False
        
        # Download person dataset
        print("\nDownloading person detection dataset...")
        try:
            project = rf.workspace("roboflow-universe-projects").project("person-detection-o4rdr")
            dataset = project.version(2).download("yolov8", location=str(raw_data_dir / "person_dataset"))
            print("Person dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading person dataset: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("DATASETS DOWNLOADED SUCCESSFULLY!")
        print("=" * 60)
        return True
        
    except ImportError:
        print("\nRoboflow library not installed. Please install it:")
        print("pip install roboflow")
        return False
    except Exception as e:
        print(f"\nError using Roboflow API: {e}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)
    
    print("\nChoose download method:")
    print("1. Use Roboflow API (automatic, requires free account)")
    print("2. Manual download instructions")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = use_roboflow_api()
        if success:
            print("\nNext step: Run 'python prepare_yolo_dataset.py'")
            return
    
    # Manual download instructions
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    
    fire_dir = download_fire_dataset_sample()
    person_dir = download_person_dataset_sample()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Download datasets using the links above")
    print("2. Extract them to the specified directories")
    print("3. Run: python prepare_yolo_dataset.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
