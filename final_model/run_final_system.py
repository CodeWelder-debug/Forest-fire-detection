from ultralytics import YOLO
import cv2
import numpy as np
import os

def run_final_system():
    # Paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m1_path = os.path.join(base_dir, "fire and person.pt")
    m2_path = os.path.join(base_dir, "aerial_images.pt")
    
    # Check if files exist
    if not os.path.exists(m1_path) or not os.path.exists(m2_path):
        print(f"Models not found in {base_dir}")
        print(f"Looking for: {m1_path}")
        print(f"Looking for: {m2_path}")
        return

    print(f"Loading M1: {m1_path}")
    model1 = YOLO(m1_path)
    print(f"Loading M2: {m2_path}")
    model2 = YOLO(m2_path)
    
    # Color mapping (BGR for OpenCV)
    # Person: Green (0, 255, 0)
    # Fire: Red (0, 0, 255)
    # Animal: Cyan (255, 255, 0) - Blue+Green
    colors = {
        'person': (0, 255, 0),
        'fire': (0, 0, 255),
        'animal': (255, 255, 0)
    }
    
    source = 1
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open camera {source}")
        return

    print("Running Final System... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results1 = model1(frame, verbose=False, conf=0.4)
        results2 = model2(frame, verbose=False, conf=0.4)
        
        all_boxes = []
        all_confidences = []
        all_labels = [] # Stores (label_name, color)
        
        # Helper to process results
        def process_results(results, model):
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name (normalize to lowercase)
                    cls_name = model.names[cls].lower()
                    
                    # Determine color
                    color = (255, 255, 255) # Default white
                    if 'person' in cls_name:
                        color = colors['person']
                        cls_name = 'Person'
                    elif 'fire' in cls_name:
                        color = colors['fire']
                        cls_name = 'Fire'
                    elif 'animal' in cls_name:
                        color = colors['animal']
                        cls_name = 'Animal'
                    
                    label_str = f"{cls_name} {conf:.2f}"
                    
                    all_boxes.append([x1, y1, w, h])
                    all_confidences.append(conf)
                    all_labels.append((label_str, color))

        process_results(results1, model1)
        process_results(results2, model2)
        
        # GLOBAL NMS
        # Merges overlaps regardless of which model found it
        if len(all_boxes) > 0:
            # iou_threshold=0.45 means merge if overlaps significantly
            indices = cv2.dnn.NMSBoxes(all_boxes, all_confidences, 0.4, 0.45)
            
            if len(indices) > 0:
                indices = indices.flatten()
                for i in indices:
                    box = all_boxes[i]
                    x1, y1, w, h = box
                    x2, y2 = x1 + w, y1 + h
                    
                    label_text, color = all_labels[i]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Final System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_final_system()
