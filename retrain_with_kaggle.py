"""
Retrain the model with additional Kaggle fire dataset.
This will enhance fire detection by adding more training data.
"""

from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
import random
from collections import defaultdict

def prepare_combined_dataset():
    """Combine existing dataset with Kaggle fire data."""
    print("=" * 60)
    print("PREPARING ENHANCED DATASET")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    # Existing dataset
    existing_dataset = base_dir / "dataset"
    
    # Kaggle dataset
    kaggle_dataset = base_dir / "kaggle_fire_dataset"
    
    # New combined dataset
    enhanced_dataset = base_dir / "enhanced_dataset"
    
    if not kaggle_dataset.exists():
        print(f"\n❌ Kaggle dataset not found at {kaggle_dataset}")
        print("Please run: python download_kaggle_fire.py")
        return None
    
    print(f"\n📊 Existing dataset: {existing_dataset}")
    print(f"📊 Kaggle dataset: {kaggle_dataset}")
    print(f"📊 Enhanced dataset: {enhanced_dataset}")
    
    # Create enhanced dataset structure
    for split in ['train', 'val']:
        (enhanced_dataset / 'images' / split).mkdir(parents=True, exist_ok=True)
        (enhanced_dataset / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("\n📋 Copying existing dataset...")
    
    # Copy existing dataset
    for split in ['train', 'val']:
        existing_images = existing_dataset / 'images' / split
        existing_labels = existing_dataset / 'labels' / split
        
        if existing_images.exists():
            for img in existing_images.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img, enhanced_dataset / 'images' / split / img.name)
            
            for lbl in existing_labels.glob('*.txt'):
                shutil.copy2(lbl, enhanced_dataset / 'labels' / split / lbl.name)
    
    print("✅ Existing dataset copied")
    
    # Process Kaggle dataset
    print("\n📋 Processing Kaggle fire images...")
    
    # Find fire images in Kaggle dataset
    fire_images = []
    for img_path in kaggle_dataset.rglob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Check if it's a fire image (usually in 'fire' folder)
            if 'fire' in str(img_path).lower() and 'non' not in str(img_path).lower():
                fire_images.append(img_path)
    
    print(f"Found {len(fire_images)} fire images in Kaggle dataset")
    
    if len(fire_images) == 0:
        print("⚠️  No fire images found in Kaggle dataset")
        print("Checking all images...")
        fire_images = list(kaggle_dataset.rglob('*.jpg'))
        fire_images.extend(kaggle_dataset.rglob('*.png'))
        print(f"Found {len(fire_images)} total images")
    
    # Split Kaggle data 80/20
    random.shuffle(fire_images)
    split_idx = int(len(fire_images) * 0.8)
    kaggle_train = fire_images[:split_idx]
    kaggle_val = fire_images[split_idx:]
    
    print(f"  Train: {len(kaggle_train)} images")
    print(f"  Val: {len(kaggle_val)} images")
    
    # Copy Kaggle images and create labels
    def process_kaggle_images(images, split):
        for i, img_path in enumerate(images):
            # Copy image
            new_img_name = f"kaggle_fire_{i:05d}{img_path.suffix}"
            new_img_path = enhanced_dataset / 'images' / split / new_img_name
            shutil.copy2(img_path, new_img_path)
            
            # Create label (assume entire image contains fire)
            # YOLO format: class x_center y_center width height (normalized)
            label_path = enhanced_dataset / 'labels' / split / f"kaggle_fire_{i:05d}.txt"
            with open(label_path, 'w') as f:
                # Class 0 (fire), centered, full image
                f.write("0 0.5 0.5 0.9 0.9\n")
    
    print("\n📋 Adding Kaggle images to training set...")
    process_kaggle_images(kaggle_train, 'train')
    process_kaggle_images(kaggle_val, 'val')
    
    print("✅ Kaggle images added")
    
    # Create data.yaml
    data_yaml = {
        'path': str(enhanced_dataset.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['fire', 'person']
    }
    
    yaml_path = enhanced_dataset / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✅ Created configuration: {yaml_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("ENHANCED DATASET STATISTICS")
    print("=" * 60)
    
    stats = defaultdict(lambda: {'train': 0, 'val': 0})
    
    for split in ['train', 'val']:
        img_count = len(list((enhanced_dataset / 'images' / split).glob('*')))
        lbl_count = len(list((enhanced_dataset / 'labels' / split).glob('*.txt')))
        print(f"\n{split.upper()}:")
        print(f"  Images: {img_count}")
        print(f"  Labels: {lbl_count}")
    
    return enhanced_dataset

def retrain_model(dataset_path):
    """Retrain the model with enhanced dataset."""
    print("\n" + "=" * 60)
    print("RETRAINING MODEL WITH ENHANCED DATASET")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    # Load existing model weights
    existing_model = base_dir / "runs" / "detect" / "runs" / "detect" / "fire_person_detection" / "weights" / "best.pt"
    
    if not existing_model.exists():
        print("⚠️  Existing model not found, starting from scratch")
        model = YOLO('yolov8n.pt')
    else:
        print(f"✅ Loading existing model: {existing_model}")
        model = YOLO(str(existing_model))
    
    data_yaml = dataset_path / 'data.yaml'
    
    print(f"\nDataset config: {data_yaml}")
    print("\nTraining configuration:")
    print("  Model: YOLOv8n (continuing from existing weights)")
    print("  Epochs: 20 (fine-tuning)")
    print("  Image size: 640")
    print("  Batch size: 16")
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=20,  # Fewer epochs for fine-tuning
        imgsz=640,
        batch=16,
        name='fire_person_enhanced',
        patience=10,
        save=True,
        device='0',
        workers=4,
        project='runs/detect',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        val=True,
        plots=True,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Validate
    print("\nValidating enhanced model...")
    metrics = model.val()
    
    print("\nEnhanced Model Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    print("\n✅ Enhanced model saved to: runs/detect/fire_person_enhanced/weights/best.pt")
    
    return model

def main():
    """Main function."""
    print("=" * 60)
    print("ENHANCE FIRE DETECTION MODEL")
    print("=" * 60)
    
    # Prepare enhanced dataset
    dataset_path = prepare_combined_dataset()
    
    if dataset_path is None:
        return
    
    # Ask user if they want to retrain
    print("\n" + "=" * 60)
    choice = input("\nDo you want to retrain the model now? (y/n): ").strip().lower()
    
    if choice == 'y':
        retrain_model(dataset_path)
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print("\nYour enhanced model is ready!")
        print("\nTo use the enhanced model for webcam:")
        print("  1. Update webcam_inference.py to use:")
        print("     runs/detect/fire_person_enhanced/weights/best.pt")
        print("\nTo test on fire images:")
        print("  python test_fire_detection.py kaggle_fire_dataset/")
    else:
        print("\nDataset prepared. Run this script again when ready to train.")

if __name__ == "__main__":
    random.seed(42)
    main()
