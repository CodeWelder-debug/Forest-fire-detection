# 🔥 Enhanced Fire and Person Detection Model

Real-time fire and person detection using YOLOv8 with enhanced fire detection accuracy.

## 📊 Model Performance

- **Overall mAP50**: 93.56%
- **Fire Detection mAP50**: 94.3%
- **Person Detection mAP50**: 92.8%
- **Precision**: 92.63%
- **Recall**: 87.15%
- **Model Size**: 6.2 MB
- **Inference Speed**: ~123 FPS

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Webcam Detection

```bash
# Default (camera index 1)
python webcam_detection.py

# Specify camera
python webcam_detection.py --camera 0

# Adjust confidence threshold
python webcam_detection.py --conf 0.5
```

### Controls

- Press `q` to quit
- Press `s` to save screenshot

## 📁 Files

- `best.pt` - Enhanced YOLOv8n model weights (6.2 MB)
- `webcam_detection.py` - Real-time detection script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🎯 Classes

- **Class 0**: Fire (Red bounding boxes)
- **Class 1**: Person (Green bounding boxes)

## 🔧 Requirements

- Python 3.8+
- Webcam
- GPU recommended (works on CPU)

## 📖 Usage Examples

### Basic Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run inference on image
results = model('image.jpg')

# Run inference on video
results = model('video.mp4')
```

### Custom Inference

```python
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    results = model(frame, conf=0.25)
    
    # Process results
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🎓 Training Details

- **Architecture**: YOLOv8n (Nano)
- **Training Images**: 2,863 (1,244 fire + 1,770 person)
- **Validation Images**: 716
- **Epochs**: 20 (fine-tuned from base model)
- **Image Size**: 640x640
- **Batch Size**: 16

## 📝 Model Info

This model was trained on a combination of:
- Roboflow fire detection dataset (612 images)
- Roboflow person detection dataset (2,212 images)
- Kaggle fire dataset (755 images)

The model achieves excellent fire detection performance (94.3% mAP50) while maintaining strong person detection capabilities.

## 🔗 Repository

For full training code and dataset preparation scripts, visit:
https://github.com/CodeWelder-debug/Forest-fire-detection

## 📄 License

Model weights and code are provided for educational and research purposes.

---

**Built with YOLOv8 by Ultralytics**
