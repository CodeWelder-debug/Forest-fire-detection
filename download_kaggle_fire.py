"""
Download Kaggle fire dataset and enhance the existing model.
This will add more fire training data to improve fire detection.
"""

import kagglehub
from pathlib import Path
import shutil
import os

def download_kaggle_dataset():
    """Download the Kaggle fire dataset."""
    print("=" * 60)
    print("DOWNLOADING KAGGLE FIRE DATASET")
    print("=" * 60)
    
    print("\n📥 Downloading from Kaggle...")
    print("Dataset: phylake1337/fire-dataset")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("phylake1337/fire-dataset")
        print(f"\n✅ Dataset downloaded!")
        print(f"Path to dataset files: {path}")
        
        return path
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed kagglehub: pip install kagglehub")
        print("2. Set up Kaggle API credentials")
        print("   - Go to https://www.kaggle.com/settings")
        print("   - Create API token")
        print("   - Place kaggle.json in ~/.kaggle/")
        return None

def prepare_kaggle_data(kaggle_path):
    """Prepare Kaggle dataset for training."""
    print("\n" + "=" * 60)
    print("PREPARING KAGGLE DATASET")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    kaggle_dir = base_dir / "kaggle_fire_dataset"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_path = Path(kaggle_path)
    
    print(f"\nKaggle data location: {kaggle_path}")
    print(f"Copying to: {kaggle_dir}")
    
    # List contents
    print("\nDataset contents:")
    for item in kaggle_path.rglob('*'):
        if item.is_file():
            print(f"  {item.relative_to(kaggle_path)}")
    
    # Copy dataset
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    shutil.copytree(kaggle_path, kaggle_dir)
    
    print(f"\n✅ Dataset copied to: {kaggle_dir}")
    
    return kaggle_dir

def main():
    """Main function."""
    print("=" * 60)
    print("KAGGLE FIRE DATASET DOWNLOADER")
    print("=" * 60)
    
    # Check if kagglehub is installed
    try:
        import kagglehub
    except ImportError:
        print("\n❌ kagglehub not installed!")
        print("\nInstalling kagglehub...")
        os.system("pip install kagglehub")
        import kagglehub
    
    # Download dataset
    kaggle_path = download_kaggle_dataset()
    
    if kaggle_path:
        # Prepare dataset
        dataset_dir = prepare_kaggle_data(kaggle_path)
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\nOption 1: Retrain existing model with more fire data")
        print("  python retrain_with_kaggle.py")
        print("\nOption 2: Test current model on Kaggle fire images")
        print("  python test_fire_detection.py kaggle_fire_dataset/")
        print("=" * 60)
    else:
        print("\n❌ Failed to download dataset")

if __name__ == "__main__":
    main()
