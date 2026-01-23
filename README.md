# Forest Fire Detection

A real-time fire and person detection system I built using YOLOv8. The model can detect fires with 94.3% accuracy and people with 92.8% accuracy.

## What it does

This project uses a trained YOLOv8 model to detect fires and people in real-time through your webcam. When it spots a fire, it draws a red box around it. People get green boxes.

I trained this on about 3,000 images - a mix of fire images from Kaggle and Roboflow, plus person detection data. The model is pretty lightweight (just 6.2 MB) and runs fast enough for real-time detection.

## Getting started

Clone this repo and install the requirements:

```bash
git clone https://github.com/CodeWelder-debug/Forest-fire-detection.git
cd Forest-fire-detection/fire_model
pip install -r requirements.txt
```

Then run the detection:

```bash
python webcam_detection.py
```

Press 'q' to quit or 's' to save a screenshot.

## How well does it work?

After training on 2,863 images, here's what I got:
- Fire detection: 94.3% mAP50
- Person detection: 92.8% mAP50
- Overall: 93.56% mAP50

The model runs at about 123 FPS, so it's smooth for real-time use.

## Using the model

If you want to use it in your own code:

```python
from ultralytics import YOLO

model = YOLO('fire_model/best.pt')

# On an image
results = model('your_image.jpg')

# On a video
results = model('your_video.mp4')
```

## What's in the fire_model folder

- `best.pt` - The trained model weights
- `webcam_detection.py` - Script for real-time detection
- `requirements.txt` - What you need to install
- `README.md` - More details about the model

## Options

You can tweak the webcam script:

```bash
# Use a different camera
python webcam_detection.py --camera 0

# Change confidence threshold
python webcam_detection.py --conf 0.5
```

## Requirements

- Python 3.8 or newer
- A webcam (for real-time detection)
- GPU helps but isn't required

## Training details

I used YOLOv8n (the nano version) because it's fast and still accurate enough. Trained for 20 epochs on:
- 1,244 fire images
- 1,770 person images

The fire detection improved a lot after I added more fire images from Kaggle to the original Roboflow dataset.

## License

Feel free to use this for learning or research.

---

Built with YOLOv8 by Ultralytics
