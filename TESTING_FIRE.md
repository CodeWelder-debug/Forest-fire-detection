# 🔥 Testing Fire Detection on Forest Fires

Your model is trained and ready! Here's how to test it on actual forest fire images/videos:

## 📥 Get Forest Fire Test Data

### Option 1: Download Sample Images
Visit these sites to download forest fire images:
- **Kaggle**: https://www.kaggle.com/datasets/phylake1337/fire-dataset
- **Google Images**: Search "forest fire" and download images
- **Pexels**: https://www.pexels.com/search/forest%20fire/
- **Unsplash**: https://unsplash.com/s/photos/wildfire

### Option 2: Download YouTube Videos
Use `yt-dlp` or online downloaders:
```bash
# Install yt-dlp
pip install yt-dlp

# Download a forest fire video
yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID" -f mp4
```

## 🧪 Test Fire Detection

### Test on a Single Image
```bash
python test_fire_detection.py path/to/fire_image.jpg
```

### Test on a Video
```bash
python test_fire_detection.py path/to/fire_video.mp4
```

### Test on a Folder of Images
```bash
python test_fire_detection.py path/to/images_folder/
```

### Save Detection Results
```bash
# Save image with detections
python test_fire_detection.py fire.jpg --save output.jpg

# Save video with detections
python test_fire_detection.py fire.mp4 --save output.mp4

# Save all images in folder
python test_fire_detection.py images/ --save output_folder/
```

### Adjust Confidence Threshold
```bash
# Higher confidence (fewer detections, more accurate)
python test_fire_detection.py fire.jpg --conf 0.5

# Lower confidence (more detections, may include false positives)
python test_fire_detection.py fire.jpg --conf 0.15
```

## 🎯 Expected Results

Your model was trained on 612 fire images and achieved:
- **Fire Detection mAP50**: 89.2%
- **Precision**: 86.5%
- **Recall**: 80.7%

This means:
- ✅ It will detect most fires in images
- ✅ Red bounding boxes will appear around fire
- ✅ Confidence scores will be displayed
- ⚠️ Some small or distant fires may be missed
- ⚠️ Very bright objects might occasionally be detected as fire

## 💡 Tips for Best Results

1. **Use clear images**: Better quality = better detection
2. **Adjust confidence**: Lower threshold (0.15-0.25) for more detections
3. **Test various scenarios**: Day/night, close/far, small/large fires
4. **Compare with person detection**: The model detects both fire and people

## 🚀 Quick Test

Want to test right now? Try this:

1. Download a forest fire image from Google Images
2. Save it as `test_fire.jpg` in the Forest fire folder
3. Run:
```bash
python test_fire_detection.py test_fire.jpg
```

The detection window will show the image with red boxes around detected fire!

## 📊 Understanding Results

When you run the test, you'll see:
```
🔥 Fire detections: 3
👤 Person detections: 0
```

- **Fire detections**: Number of fire instances found
- **Person detections**: Number of people found
- Each detection shows confidence score (0.0-1.0)

## 🎥 Video Detection

For videos, the script will:
- Process each frame
- Draw bounding boxes in real-time
- Show total detections at the end
- Optionally save the annotated video

Press `q` to stop video processing early.

---

**Your model is ready to detect forest fires! Download some test images/videos and try it out!** 🔥
