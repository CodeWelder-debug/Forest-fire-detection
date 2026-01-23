"""
Quick test to verify webcam and model are ready.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

def test_webcam_and_model():
    """Test webcam access and model loading."""
    print("=" * 60)
    print("WEBCAM AND MODEL TEST")
    print("=" * 60)
    
    # Check if model exists
    base_dir = Path(__file__).parent
    model_path = base_dir / "runs" / "detect" / "runs" / "detect" / "fire_person_detection" / "weights" / "best.pt"
    
    print("\n1. Checking model...")
    if model_path.exists():
        print(f"   ✅ Model found: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"   ❌ Model not found at {model_path}")
        return False
    
    # Load model
    print("\n2. Loading model...")
    try:
        model = YOLO(str(model_path))
        print("   ✅ Model loaded successfully!")
        print(f"   Classes: {model.names}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False
    
    # Check webcam
    print("\n3. Checking webcam access...")
    print("   Testing camera indices 0-3...")
    
    available_cameras = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"   ✅ Camera {i}: Available ({width}x{height})")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"   ❌ Camera {i}: Not available")
    
    if not available_cameras:
        print("\n   ⚠️  No cameras found!")
        print("   Please check:")
        print("   - Camera is connected")
        print("   - Camera permissions are granted")
        print("   - Camera is not being used by another application")
        return False
    
    print(f"\n   Found {len(available_cameras)} camera(s): {available_cameras}")
    
    if 1 in available_cameras:
        print(f"   ✅ Camera index 1 is available (as requested)")
    else:
        print(f"   ⚠️  Camera index 1 not available")
        print(f"   Available cameras: {available_cameras}")
        print(f"   You can use camera index {available_cameras[0]} instead")
    
    print("\n" + "=" * 60)
    print("SYSTEM READY!")
    print("=" * 60)
    print("\nEverything is set up correctly!")
    print("\nTo start real-time detection:")
    print("  python webcam_inference.py")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_webcam_and_model()
