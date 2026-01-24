"""
Enhanced Fire and Person Detection - Webcam Inference (GPU Optimized)
Model: YOLOv8n Enhanced (94.3% Fire mAP50, 93.56% Overall mAP50)
"""

import cv2
from ultralytics import YOLO
import time

def run_detection(model_path="best.pt", camera_index=0, conf_threshold=0.25, frame_size=(640, 480)):
    """
    Run real-time fire and person detection on webcam using GPU.
    
    Args:
        model_path: Path to YOLO model weights
        camera_index: Index of camera (default: 0)
        conf_threshold: Confidence threshold (default: 0.25)
        frame_size: Tuple (width, height) to resize frames
    """
    print("=" * 70)
    print("ENHANCED FIRE AND PERSON DETECTION".center(70))
    print("=" * 70)
    
    # Load YOLO model
    print(f"\n✅ Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Model loaded! Classes: {model.names}")
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"⚡ Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = "cuda"
    else:
        print("⚠️ CUDA not available. Using CPU.")
        device = "cpu"
    
    # Class colors
    colors = {0: (0, 0, 255), 1: (0, 255, 0)}
    
    # Open webcam
    print(f"\n📹 Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}. Trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No cameras available!")
            return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera opened: {width}x{height}")
    
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    screenshot_count = 0
    
    print("\nPress 'q' to quit, 's' to save screenshot\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for smoother GPU inference
            frame_resized = cv2.resize(frame, frame_size)
            
            # Run YOLO on GPU
            results = model(frame_resized, conf=conf_threshold, device=device, verbose=False)
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    label = f"{model.names[cls]}: {conf:.2f}"
                    color = colors.get(cls, (255, 255, 255))
                    
                    # Draw box and label
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 3)
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame_resized, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
                    cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_display = 30 / (time.time() - start_time)
                start_time = time.time()
            
            # Display FPS
            cv2.putText(frame_resized, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame_resized, "Enhanced YOLOv8 Model", (10, frame_size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Fire and Person Detection", frame_resized)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"detection_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame_resized)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Processed {frame_count} frames, {screenshot_count} screenshots saved")

# Run script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fire and Person Detection')
    parser.add_argument('--model', default='best.pt', help='Path to model weights')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    run_detection(args.model, args.camera, args.conf)
