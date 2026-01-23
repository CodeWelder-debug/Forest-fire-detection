"""
Download fire and person detection datasets from public sources.
This script downloads datasets in YOLO format from Roboflow Universe.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print("\nDownload complete!")

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

def download_roboflow_dataset(workspace, project, version, api_key, output_dir, dataset_name):
    """Download dataset from Roboflow."""
    try:
        from roboflow import Roboflow
        
        print(f"\nDownloading {dataset_name} from Roboflow...")
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8", location=output_dir)
        print(f"{dataset_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        return False

def download_datasets_alternative():
    """
    Alternative method: Download datasets using direct URLs.
    This uses publicly available datasets that don't require API keys.
    """
    base_dir = Path(__file__).parent
    raw_data_dir = base_dir / "raw_datasets"
    raw_data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ALTERNATIVE DOWNLOAD METHOD")
    print("=" * 60)
    print("\nUsing publicly available datasets from Kaggle/GitHub...")
    
    # We'll create a simple dataset structure manually
    # For fire dataset, we'll use a public fire detection dataset
    # For person dataset, we'll use COCO subset
    
    fire_dir = raw_data_dir / "fire_dataset"
    person_dir = raw_data_dir / "person_dataset"
    
    fire_dir.mkdir(exist_ok=True)
    person_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("IMPORTANT: Manual Dataset Setup Required")
    print("=" * 60)
    print("\nSince we need public datasets, please follow these steps:")
    print("\n1. Fire Dataset:")
    print("   - Visit: https://www.kaggle.com/datasets/phylake1337/fire-dataset")
    print("   - Or: https://universe.roboflow.com/fire-detection/fire-detection-dataset")
    print("   - Download and extract to: raw_datasets/fire_dataset/")
    print("\n2. Person Dataset:")
    print("   - Visit: https://universe.roboflow.com/coco-dataset/coco-person")
    print("   - Or use COCO dataset person subset")
    print("   - Download and extract to: raw_datasets/person_dataset/")
    print("\n3. Expected structure:")
    print("   raw_datasets/")
    print("   ├── fire_dataset/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   └── person_dataset/")
    print("       ├── images/")
    print("       └── labels/")
    print("\nAlternatively, I'll create a script to use Roboflow API...")
    print("=" * 60)
    
    return fire_dir, person_dir

def main():
    """Main function to download datasets."""
    base_dir = Path(__file__).parent
    raw_data_dir = base_dir / "raw_datasets"
    raw_data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("FIRE AND PERSON DATASET DOWNLOADER")
    print("=" * 60)
    
    # Try to use Roboflow API
    print("\nAttempting to download datasets using Roboflow...")
    print("\nTo use Roboflow, you need an API key (free):")
    print("1. Visit: https://app.roboflow.com/")
    print("2. Sign up for a free account")
    print("3. Get your API key from Settings")
    
    api_key = input("\nEnter your Roboflow API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Download fire dataset
        fire_success = download_roboflow_dataset(
            workspace="fire-detection-jlqmm",
            project="fire-detection-nnvxp",
            version=1,
            api_key=api_key,
            output_dir=str(raw_data_dir / "fire_dataset"),
            dataset_name="Fire Detection Dataset"
        )
        
        # Download person dataset
        person_success = download_roboflow_dataset(
            workspace="coco-dataset",
            project="coco-person",
            version=1,
            api_key=api_key,
            output_dir=str(raw_data_dir / "person_dataset"),
            dataset_name="Person Detection Dataset"
        )
        
        if fire_success and person_success:
            print("\n" + "=" * 60)
            print("SUCCESS! All datasets downloaded.")
            print("=" * 60)
            print(f"\nDatasets saved to: {raw_data_dir}")
            print("\nNext step: Run prepare_yolo_dataset.py to combine datasets")
            return
    
    # Alternative method
    print("\n" + "=" * 60)
    print("Using alternative download method...")
    print("=" * 60)
    download_datasets_alternative()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Download datasets manually (see instructions above)")
    print("2. Or re-run this script with a Roboflow API key")
    print("3. Then run: python prepare_yolo_dataset.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
