"""
Prepare combined fire and person detection dataset in YOLO format.
This script combines fire and person datasets, remaps labels, and creates train/val splits.
"""

import os
import shutil
import yaml
import random
from pathlib import Path
from collections import defaultdict

def find_dataset_structure(dataset_dir):
    """
    Find images and labels directories in the dataset.
    Handles various dataset structures.
    """
    dataset_path = Path(dataset_dir)
    
    # Common YOLO dataset structures
    possible_structures = [
        # Structure 1: train/val with images and labels
        {
            'train_images': dataset_path / 'train' / 'images',
            'train_labels': dataset_path / 'train' / 'labels',
            'val_images': dataset_path / 'valid' / 'images',
            'val_labels': dataset_path / 'valid' / 'labels',
        },
        # Structure 2: direct images and labels folders
        {
            'train_images': dataset_path / 'images' / 'train',
            'train_labels': dataset_path / 'labels' / 'train',
            'val_images': dataset_path / 'images' / 'val',
            'val_labels': dataset_path / 'labels' / 'val',
        },
        # Structure 3: flat images and labels
        {
            'images': dataset_path / 'images',
            'labels': dataset_path / 'labels',
        },
    ]
    
    for structure in possible_structures:
        if all(path.exists() for path in structure.values()):
            return structure
    
    # If no standard structure found, search for any images/labels directories
    images_dirs = list(dataset_path.rglob('images'))
    labels_dirs = list(dataset_path.rglob('labels'))
    
    if images_dirs and labels_dirs:
        return {
            'images': images_dirs[0],
            'labels': labels_dirs[0],
        }
    
    return None

def collect_dataset_files(dataset_dir, class_id):
    """
    Collect all images and labels from a dataset directory.
    Returns list of (image_path, label_path, class_id) tuples.
    """
    structure = find_dataset_structure(dataset_dir)
    
    if not structure:
        print(f"Warning: Could not find valid dataset structure in {dataset_dir}")
        return []
    
    files = []
    
    # Handle different structures
    if 'train_images' in structure:
        # Structure with train/val splits
        for split in ['train', 'val']:
            img_key = f'{split}_images'
            lbl_key = f'{split}_labels'
            
            if img_key in structure and lbl_key in structure:
                img_dir = structure[img_key]
                lbl_dir = structure[lbl_key]
                
                if img_dir.exists():
                    for img_file in img_dir.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            label_file = lbl_dir / f"{img_file.stem}.txt"
                            if label_file.exists():
                                files.append((img_file, label_file, class_id))
    else:
        # Flat structure
        img_dir = structure['images']
        lbl_dir = structure['labels']
        
        if img_dir.exists():
            for img_file in img_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_file = lbl_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        files.append((img_file, label_file, class_id))
    
    return files

def remap_labels(label_path, new_class_id):
    """
    Read a YOLO label file and remap all class IDs to new_class_id.
    Returns list of remapped label lines.
    """
    remapped_lines = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    # YOLO format: class_id x_center y_center width height
                    parts[0] = str(new_class_id)
                    remapped_lines.append(' '.join(parts))
    
    return remapped_lines

def create_combined_dataset(fire_dir, person_dir, output_dir, train_split=0.8):
    """
    Combine fire and person datasets into a unified YOLO dataset.
    
    Args:
        fire_dir: Path to fire dataset
        person_dir: Path to person dataset
        output_dir: Path to output combined dataset
        train_split: Ratio of training data (default 0.8 = 80%)
    """
    output_path = Path(output_dir)
    
    # Create output directory structure
    train_img_dir = output_path / 'images' / 'train'
    train_lbl_dir = output_path / 'labels' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    val_lbl_dir = output_path / 'labels' / 'val'
    
    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COMBINING DATASETS")
    print("=" * 60)
    
    # Collect files from both datasets
    print("\nCollecting fire dataset files...")
    fire_files = collect_dataset_files(fire_dir, class_id=0)  # fire = 0
    print(f"Found {len(fire_files)} fire images")
    
    print("\nCollecting person dataset files...")
    person_files = collect_dataset_files(person_dir, class_id=1)  # person = 1
    print(f"Found {len(person_files)} person images")
    
    # Combine and shuffle
    all_files = fire_files + person_files
    random.shuffle(all_files)
    
    print(f"\nTotal images: {len(all_files)}")
    
    if len(all_files) == 0:
        print("\nERROR: No valid image-label pairs found!")
        print("Please check your dataset directories:")
        print(f"  Fire dataset: {fire_dir}")
        print(f"  Person dataset: {person_dir}")
        return False
    
    # Split into train and validation
    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Process training files
    print("\nProcessing training set...")
    process_files(train_files, train_img_dir, train_lbl_dir)
    
    # Process validation files
    print("Processing validation set...")
    process_files(val_files, val_img_dir, val_lbl_dir)
    
    # Create data.yaml
    create_data_yaml(output_path)
    
    # Print statistics
    print_dataset_stats(output_path)
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nDataset saved to: {output_path}")
    print(f"Configuration file: {output_path / 'data.yaml'}")
    
    return True

def process_files(files, img_dir, lbl_dir):
    """Process and copy files to destination directories."""
    for idx, (img_path, lbl_path, class_id) in enumerate(files):
        # Create unique filename
        new_name = f"{class_id}_{idx:05d}"
        
        # Copy image
        new_img_path = img_dir / f"{new_name}{img_path.suffix}"
        shutil.copy2(img_path, new_img_path)
        
        # Remap and copy label
        remapped_labels = remap_labels(lbl_path, class_id)
        new_lbl_path = lbl_dir / f"{new_name}.txt"
        
        with open(new_lbl_path, 'w') as f:
            f.write('\n'.join(remapped_labels))
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(files)} files...")

def create_data_yaml(output_dir):
    """Create YOLO data.yaml configuration file."""
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['fire', 'person']
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\nCreated configuration file: {yaml_path}")

def print_dataset_stats(output_dir):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    stats = defaultdict(lambda: {'train': 0, 'val': 0})
    
    for split in ['train', 'val']:
        lbl_dir = output_dir / 'labels' / split
        
        for lbl_file in lbl_dir.glob('*.txt'):
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_name = 'fire' if class_id == 0 else 'person'
                        stats[class_name][split] += 1
    
    print("\nClass distribution:")
    print(f"  Fire   - Train: {stats['fire']['train']:4d}, Val: {stats['fire']['val']:4d}")
    print(f"  Person - Train: {stats['person']['train']:4d}, Val: {stats['person']['val']:4d}")
    print(f"  Total  - Train: {stats['fire']['train'] + stats['person']['train']:4d}, "
          f"Val: {stats['fire']['val'] + stats['person']['val']:4d}")

def main():
    """Main function."""
    base_dir = Path(__file__).parent
    
    # Define directories
    raw_data_dir = base_dir / "raw_datasets"
    fire_dir = raw_data_dir / "fire_dataset"
    person_dir = raw_data_dir / "person_dataset"
    output_dir = base_dir / "dataset"
    
    print("=" * 60)
    print("YOLO DATASET PREPARATION")
    print("=" * 60)
    
    # Check if raw datasets exist
    if not fire_dir.exists():
        print(f"\nERROR: Fire dataset not found at {fire_dir}")
        print("Please run download_datasets.py first or manually download datasets.")
        return
    
    if not person_dir.exists():
        print(f"\nERROR: Person dataset not found at {person_dir}")
        print("Please run download_datasets.py first or manually download datasets.")
        return
    
    print(f"\nFire dataset: {fire_dir}")
    print(f"Person dataset: {person_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create combined dataset
    success = create_combined_dataset(fire_dir, person_dir, output_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("Run: python train_model.py")
        print("=" * 60)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
