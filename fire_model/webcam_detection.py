"""
Enhanced Fire and Person Detection - Webcam Inference
Model: YOLOv8n Enhanced (94.3% Fire mAP50, 93.56% Overall mAP50)
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
import time

def run_detection(model_path="best.pt", camera_index=1, conf_threshold=0.25):
    """
    Run real-time fire and person detection on webcam.
    
    Args:
        model_path: Path to model weights (default: best.pt)
        camera_index: Camera index (default: 1)
        conf_threshold: Confidence threshold (default: 0.25)
    """
    print("=" * 70)
    print("ENHANCED FIRE AND PERSON DETECTION".center(70))
    print("=" * 70)
    
    # Load model
    print(f"\n✅ Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Model loaded! Classes: {model.names}")
    
    # Class colors
    colors = {
        0: (0, 0, 255),    # Fire - Red
        1: (0, 255, 0),    # Person - Green
    }
    
    # Open webcam
    print(f"\n📹 Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        print("Trying camera 0...")
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ No cameras available!")
            return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Camera opened: {width}x{height} @ {fps} FPS")
    print("\n" + "=" * 70)
    print("DETECTION ACTIVE".center(70))
    print("=" * 70)
    print("Controls: Press 'q' to quit, 's' to save screenshot\n")
    
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
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    class_name = model.names[cls]
                    color = colors.get(cls, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    
                    print(f"Detected: {class_name} ({conf:.2f})")
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_display = 30 / (time.time() - start_time)
                start_time = time.time()
            
            # Display info
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Enhanced Model (94.3% Fire mAP50)", (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Fire and Person Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"detection_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Processed {frame_count} frames, {screenshot_count} screenshots saved")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fire and Person Detection')
    parser.add_argument('--model', default='best.pt', help='Path to model weights')
    parser.add_argument('--camera', type=int, default=1, help='Camera index')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    run_detection(args.model, args.camera, args.conf)
