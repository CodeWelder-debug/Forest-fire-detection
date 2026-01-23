"""
Train YOLOv8 model - Auto mode (non-interactive).
Uses sensible defaults for quick training.
"""

import os
from pathlib import Path
from ultralytics import YOLO

def train_model_auto():
    """Train YOLOv8 model with automatic settings."""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "dataset"
    data_yaml = dataset_dir / "data.yaml"
    
    print("=" * 60)
    print("YOLOV8 MODEL TRAINING (AUTO MODE)")
    print("=" * 60)
    
    # Check if dataset exists
    if not data_yaml.exists():
        print(f"\nERROR: Dataset configuration not found at {data_yaml}")
        print("Please run prepare_yolo_dataset.py first.")
        return False
    
    print(f"\nDataset configuration: {data_yaml}")
    
    # Auto configuration
    model_size = 'n'  # Nano - fastest for testing
    epochs = 30  # Reduced for faster training
    img_size = 640
    batch_size = 16
    
    print(f"\nTraining configuration:")
    print(f"  Model: YOLOv8{model_size} (Nano)")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: Auto-detect (GPU if available)")
    
    # Load YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nLoading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    print("\nThis may take 10-30 minutes depending on your hardware.")
    print("Training progress will be displayed below.\n")
    
    # Train the model
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='fire_person_detection',
            patience=10,
            save=True,
            device='0',  # Use GPU 0 if available, otherwise CPU
            workers=4,
            project='runs/detect',
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            val=True,
            plots=True,
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        # Print results
        print("\nTraining Results:")
        print(f"  Best model: runs/detect/fire_person_detection/weights/best.pt")
        print(f"  Last model: runs/detect/fire_person_detection/weights/last.pt")
        print(f"  Results: runs/detect/fire_person_detection/")
        
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
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print("\nNext step: Run 'python webcam_inference.py'")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR DURING TRAINING")
        print("=" * 60)
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    success = train_model_auto()
    if not success:
        exit(1)
