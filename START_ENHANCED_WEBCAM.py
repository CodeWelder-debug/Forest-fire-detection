"""
START ENHANCED WEBCAM DETECTION - Fire and Person Detection System
Uses the enhanced model with improved fire detection (94.3% mAP50)
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch webcam inference with enhanced model."""
    print("=" * 70)
    print("ENHANCED FIRE DETECTION - WEBCAM INFERENCE".center(70))
    print("=" * 70)
    
    print("\n🎯 ENHANCED MODEL STATUS")
    print("-" * 70)
    print("✅ Model: Enhanced with Kaggle dataset")
    print("✅ Fire Detection: 94.3% mAP50 (improved from 89.2%)")
    print("✅ Overall Accuracy: 93.56% mAP50")
    print("✅ Camera index 1: Available")
    print("✅ Classes: fire (red), person (green)")
    
    print("\n📹 STARTING ENHANCED WEBCAM DETECTION...")
    print("-" * 70)
    print("Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save screenshot")
    print("-" * 70)
    
    # Create temporary webcam script with enhanced model
    base_dir = Path(__file__).parent
    
    # Import and modify webcam_inference
    import webcam_inference
    
    # Override the model path
    enhanced_model = base_dir / "runs" / "detect" / "runs" / "detect" / "fire_person_enhanced" / "weights" / "best.pt"
    
    if not enhanced_model.exists():
        print("\n❌ Enhanced model not found!")
        print(f"Expected at: {enhanced_model}")
        print("\nPlease run: python retrain_with_kaggle.py")
        return
    
    print(f"\n✅ Using enhanced model: {enhanced_model}")
    
    # Run inference with enhanced model
    from ultralytics import YOLO
    import cv2
    import time
    
    print("\nLoading enhanced model...")
    model = YOLO(str(enhanced_model))
    
    # Class names and colors
    class_names = ['fire', 'person']
    colors = {
        0: (0, 0, 255),    # Fire - Red
        1: (0, 255, 0),    # Person - Green
    }
    
    # Open webcam
    camera_index = 1
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"\n❌ Could not open camera {camera_index}")
        print("Trying camera 0...")
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ No cameras available!")
            return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n✅ Camera opened!")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    print("\n" + "=" * 70)
    print("DETECTION ACTIVE - Enhanced Model".center(70))
    print("=" * 70)
    
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = model(frame, conf=0.25, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    class_name = class_names[cls]
                    color = colors[cls]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Create label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Get label size
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    print(f"Detected: {class_name} (confidence: {conf:.2f})")
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps_display = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Draw FPS
            cv2.putText(
                frame,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            
            # Draw model info
            cv2.putText(
                frame,
                "ENHANCED MODEL (94.3% Fire mAP50)",
                (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            
            # Draw instructions
            cv2.putText(
                frame,
                "Press 'q' to quit, 's' to save",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Display frame
            cv2.imshow('Enhanced Fire and Person Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                screenshot_path = f"enhanced_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"\nScreenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("DETECTION STOPPED")
        print("=" * 70)
        print(f"\nTotal frames processed: {frame_count}")
        print(f"Screenshots saved: {screenshot_count}")

if __name__ == "__main__":
    main()
