"""
Test fire detection on images or videos.
This script allows you to test the fire detection model on:
- Single images
- Folders of images
- Video files
- YouTube videos (URLs)
"""

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
import argparse

def detect_on_image(model, image_path, conf_threshold=0.25, save_path=None):
    """Run detection on a single image."""
    print(f"\n📸 Processing image: {image_path}")
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    # Run inference
    results = model(img, conf=conf_threshold, verbose=False)
    
    # Process results
    detections = {'fire': 0, 'person': 0}
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            class_name = 'fire' if cls == 0 else 'person'
            color = (0, 0, 255) if cls == 0 else (0, 255, 0)
            
            detections[class_name] += 1
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    print(f"  🔥 Fire detections: {detections['fire']}")
    print(f"  👤 Person detections: {detections['person']}")
    
    # Save or display
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"  ✅ Saved to: {save_path}")
    else:
        cv2.imshow('Fire Detection', img)
        print("  Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_on_video(model, video_path, conf_threshold=0.25, save_path=None):
    """Run detection on a video file."""
    print(f"\n🎥 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Setup video writer if saving
    out = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = {'fire': 0, 'person': 0}
    
    print("\n  Processing... (Press 'q' to stop)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                class_name = 'fire' if cls == 0 else 'person'
                color = (0, 0, 255) if cls == 0 else (0, 255, 0)
                
                total_detections[class_name] += 1
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save or display
        if out:
            out.write(frame)
            if frame_count % 30 == 0:
                print(f"  Progress: {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%)")
        else:
            cv2.imshow('Fire Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\n  ✅ Processed {frame_count} frames")
    print(f"  🔥 Total fire detections: {total_detections['fire']}")
    print(f"  👤 Total person detections: {total_detections['person']}")
    if save_path:
        print(f"  💾 Saved to: {save_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test fire detection on images/videos')
    parser.add_argument('input', help='Path to image, video, or folder')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--save', help='Save output to this path')
    
    args = parser.parse_args()
    
    # Load model
    base_dir = Path(__file__).parent
    model_path = base_dir / "runs" / "detect" / "runs" / "detect" / "fire_person_detection" / "weights" / "best.pt"
    
    print("=" * 60)
    print("FIRE DETECTION TESTER")
    print("=" * 60)
    
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        return
    
    print(f"\n✅ Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"✅ Model loaded! Classes: {model.names}")
    
    input_path = Path(args.input)
    
    # Check if input exists
    if not input_path.exists():
        print(f"\n❌ Input not found: {input_path}")
        return
    
    # Process based on input type
    if input_path.is_file():
        # Check if image or video
        ext = input_path.suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            detect_on_image(model, input_path, args.conf, args.save)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            detect_on_video(model, input_path, args.conf, args.save)
        else:
            print(f"\n❌ Unsupported file type: {ext}")
    elif input_path.is_dir():
        # Process all images in folder
        print(f"\n📁 Processing folder: {input_path}")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(input_path.glob(ext))
        
        if not image_files:
            print("❌ No images found in folder")
            return
        
        print(f"Found {len(image_files)} images")
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]")
            save_path = None
            if args.save:
                save_dir = Path(args.save)
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / f"detected_{img_path.name}"
            detect_on_image(model, img_path, args.conf, save_path)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode
        print("=" * 60)
        print("FIRE DETECTION TESTER - Interactive Mode")
        print("=" * 60)
        print("\nUsage examples:")
        print("  python test_fire_detection.py image.jpg")
        print("  python test_fire_detection.py video.mp4")
        print("  python test_fire_detection.py folder_of_images/")
        print("  python test_fire_detection.py image.jpg --save output.jpg")
        print("  python test_fire_detection.py video.mp4 --save output.mp4")
        print("  python test_fire_detection.py image.jpg --conf 0.5")
        print("\nTip: Download forest fire images/videos from YouTube or Google Images to test!")
        print("=" * 60)
    else:
        main()
