"""
Train YOLOv8 model for fire and person detection.
This script trains a YOLOv8 model on the combined dataset.
"""

import os
from pathlib import Path
from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, img_size=640, batch_size=16, model_size='n'):
    """
    Train YOLOv8 model.
    
    Args:
        data_yaml_path: Path to data.yaml configuration file
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size for training
        model_size: Model size ('n', 's', 'm', 'l', 'x')
    """
    print("=" * 60)
    print("YOLOV8 MODEL TRAINING")
    print("=" * 60)
    
    # Load YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nLoading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    
    print(f"\nTraining configuration:")
    print(f"  Data config: {data_yaml_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: YOLOv8{model_size}")
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    
    # Train the model
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='fire_person_detection',
        patience=10,  # Early stopping patience
        save=True,
        device=0,  # Use GPU if available, otherwise CPU
        workers=4,
        project='runs/detect',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Print results
    print("\nTraining Results:")
    print(f"  Best model saved to: runs/detect/fire_person_detection/weights/best.pt")
    print(f"  Last model saved to: runs/detect/fire_person_detection/weights/last.pt")
    
    # Validate the model
    print("\n" + "=" * 60)
    print("VALIDATING MODEL...")
    print("=" * 60)
    
    metrics = model.val()
    
    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return model, results, metrics

def main():
    """Main function."""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "dataset"
    data_yaml = dataset_dir / "data.yaml"
    
    print("=" * 60)
    print("FIRE AND PERSON DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Check if dataset exists
    if not data_yaml.exists():
        print(f"\nERROR: Dataset configuration not found at {data_yaml}")
        print("Please run prepare_yolo_dataset.py first.")
        return
    
    print(f"\nDataset configuration: {data_yaml}")
    
    # Ask user for training parameters
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    print("\nModel sizes:")
    print("  n - Nano (fastest, least accurate)")
    print("  s - Small")
    print("  m - Medium (recommended)")
    print("  l - Large")
    print("  x - Extra Large (slowest, most accurate)")
    
    model_size = input("\nSelect model size [n/s/m/l/x] (default: n): ").strip().lower() or 'n'
    
    if model_size not in ['n', 's', 'm', 'l', 'x']:
        print(f"Invalid model size '{model_size}', using 'n'")
        model_size = 'n'
    
    epochs_input = input("Number of epochs (default: 50): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 50
    
    batch_input = input("Batch size (default: 16): ").strip()
    batch_size = int(batch_input) if batch_input.isdigit() else 16
    
    img_input = input("Image size (default: 640): ").strip()
    img_size = int(img_input) if img_input.isdigit() else 640
    
    # Train the model
    try:
        model, results, metrics = train_model(
            data_yaml_path=data_yaml,
            epochs=epochs,
            img_size=img_size,
            batch_size=batch_size,
            model_size=model_size
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print("\nModel training completed successfully!")
        print("\nNext steps:")
        print("  1. Check training results in: runs/detect/fire_person_detection/")
        print("  2. Run webcam inference: python webcam_inference.py")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR DURING TRAINING")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. Dataset is properly prepared")
        print("  2. CUDA/GPU drivers if using GPU")
        print("  3. Sufficient disk space")
        print("=" * 60)
        raise

if __name__ == "__main__":
    main()
