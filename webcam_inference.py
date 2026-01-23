"""
Real-time fire and person detection using webcam.
This script uses the trained YOLOv8 model to detect fire and person in real-time.
"""

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
import time

def run_webcam_inference(model_path, camera_index=1, conf_threshold=0.25):
    """
    Run real-time inference on webcam feed.
    
    Args:
        model_path: Path to trained model weights
        camera_index: Camera index (default: 1)
        conf_threshold: Confidence threshold for detections
    """
    print("=" * 60)
    print("REAL-TIME FIRE AND PERSON DETECTION")
    print("=" * 60)
    
    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    
    # Class names
    class_names = ['fire', 'person']
    
    # Colors for bounding boxes (BGR format)
    colors = {
        0: (0, 0, 255),    # Fire - Red
        1: (0, 255, 0),    # Person - Green
    }
    
    # Open webcam
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"\nERROR: Could not open camera {camera_index}")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected")
        print("  2. Try different camera index (0, 1, 2, etc.)")
        print("  3. Check camera permissions")
        
        # Try to list available cameras
        print("\nTrying to find available cameras...")
        for i in range(5):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                print(f"  Camera {i}: Available")
                test_cap.release()
            else:
                print(f"  Camera {i}: Not available")
        
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\nCamera opened successfully!")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Confidence threshold: {conf_threshold}")
    
    print("\n" + "=" * 60)
    print("STARTING DETECTION...")
    print("=" * 60)
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save screenshot")
    print("=" * 60)
    
    # FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    screenshot_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("\nERROR: Failed to read frame from camera")
                break
            
            # Run inference
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name and color
                    class_name = class_names[cls]
                    color = colors[cls]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Get label size
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
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
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    # Print detection to console
                    print(f"Detected: {class_name} (confidence: {conf:.2f})")
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps_display = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Draw FPS on frame
            cv2.putText(
                frame,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
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
            cv2.imshow('Fire and Person Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                screenshot_path = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"\nScreenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("DETECTION STOPPED")
        print("=" * 60)
        print(f"\nTotal frames processed: {frame_count}")
        print(f"Screenshots saved: {screenshot_count}")

def main():
    """Main function."""
    base_dir = Path(__file__).parent
    
    # Default model path
    default_model = base_dir / "runs" / "detect" / "runs" / "detect" / "fire_person_detection" / "weights" / "best.pt"
    
    print("=" * 60)
    print("WEBCAM INFERENCE - FIRE AND PERSON DETECTION")
    print("=" * 60)
    
    # Check if trained model exists
    if not default_model.exists():
        print(f"\nERROR: Trained model not found at {default_model}")
        print("\nPlease train the model first by running: python train_model.py")
        
        # Check if any model exists
        runs_dir = base_dir / "runs" / "detect"
        if runs_dir.exists():
            print("\nAvailable models:")
            for model_dir in runs_dir.iterdir():
                if model_dir.is_dir():
                    best_pt = model_dir / "weights" / "best.pt"
                    if best_pt.exists():
                        print(f"  {best_pt}")
        
        return
    
    print(f"\nModel found: {default_model}")
    
    # Get camera index
    print("\n" + "=" * 60)
    print("CAMERA CONFIGURATION")
    print("=" * 60)
    
    camera_input = input("\nEnter camera index (default: 1): ").strip()
    camera_index = int(camera_input) if camera_input.isdigit() else 1
    
    # Get confidence threshold
    conf_input = input("Enter confidence threshold 0-1 (default: 0.25): ").strip()
    try:
        conf_threshold = float(conf_input) if conf_input else 0.25
        conf_threshold = max(0.0, min(1.0, conf_threshold))
    except ValueError:
        conf_threshold = 0.25
    
    # Run inference
    run_webcam_inference(
        model_path=default_model,
        camera_index=camera_index,
        conf_threshold=conf_threshold
    )

if __name__ == "__main__":
    main()
