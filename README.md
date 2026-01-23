# 🔥 Forest Fire Detection System

Real-time fire and person detection using YOLOv8 with enhanced fire detection accuracy (94.3% mAP50).

## 🎯 Features

- **Enhanced Fire Detection**: 94.3% mAP50 accuracy
- **Person Detection**: 92.8% mAP50 accuracy
- **Real-time Webcam Inference**: ~123 FPS
- **Easy to Use**: Simple Python scripts
- **Pre-trained Model**: Ready to use out of the box

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Overall mAP50 | **93.56%** |
| Fire mAP50 | **94.3%** |
| Person mAP50 | 92.8% |
| Precision | 92.63% |
| Recall | 87.15% |
| Model Size | 6.2 MB |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/CodeWelder-debug/Forest-fire-detection.git
cd Forest-fire-detection

# Install dependencies
cd fire_model
pip install -r requirements.txt
```

### Run Detection

```bash
# Run webcam detection
python webcam_detection.py

# Specify camera index
python webcam_detection.py --camera 0

# Adjust confidence threshold
python webcam_detection.py --conf 0.5
```

## 📁 Project Structure

```
Forest-fire-detection/
├── fire_model/                    # 🔥 Main model folder
│   ├── best.pt                    # Enhanced model weights (6.2 MB)
│   ├── webcam_detection.py        # Real-time detection script
│   ├── requirements.txt           # Dependencies
│   └── README.md                  # Model documentation
├── download_datasets.py           # Dataset download scripts
├── prepare_yolo_dataset.py        # Dataset preparation
├── train_model.py                 # Training scripts
├── retrain_with_kaggle.py         # Enhanced training
├── test_fire_detection.py         # Testing utilities
└── README.md                      # This file
```

## 🎓 Training Details

The model was trained on:
- **2,863 training images** (1,244 fire + 1,770 person)
- **716 validation images**
- **Datasets**: Roboflow + Kaggle fire datasets
- **Architecture**: YOLOv8n (Nano)
- **Training**: 20 epochs fine-tuning

## 📖 Usage Examples

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO('fire_model/best.pt')

# Detect on image
results = model('image.jpg')

# Detect on video
results = model('video.mp4')

# Real-time webcam
results = model(source=1, show=True)
```

### Command Line

```bash
# Detect on image
yolo predict model=fire_model/best.pt source=image.jpg

# Detect on video
yolo predict model=fire_model/best.pt source=video.mp4

# Webcam
yolo predict model=fire_model/best.pt source=1 show=True
```

## 🎯 Detection Classes

- **Fire** 🔥: Red bounding boxes
- **Person** 👤: Green bounding boxes

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Webcam (for real-time detection)

## 📝 Model Information

- **Input Size**: 640x640
- **Parameters**: 3.2M
- **GFLOPs**: 8.1
- **Inference Speed**: ~8ms per image (~123 FPS)

## 🎥 Demo

Run the webcam detection script to see real-time fire and person detection:

```bash
cd fire_model
python webcam_detection.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

## 📚 Documentation

For detailed documentation, training procedures, and dataset preparation:
- See `fire_model/README.md` for model usage
- See training scripts for custom training
- See `TESTING_FIRE.md` for testing guidelines

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is provided for educational and research purposes.

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **Roboflow** for dataset hosting
- **Kaggle** for fire dataset

---

**Built with ❤️ using YOLOv8**

For questions or issues, please open an issue on GitHub.
