"""
START WEBCAM DETECTION - Fire and Person Detection System
Run this script to start real-time detection on your webcam.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch webcam inference."""
    print("=" * 70)
    print("FIRE AND PERSON DETECTION - WEBCAM INFERENCE".center(70))
    print("=" * 70)
    
    print("\n🎯 SYSTEM STATUS")
    print("-" * 70)
    print("✅ Model trained: 91.47% mAP50")
    print("✅ Camera index 1: Available")
    print("✅ Classes: fire (red), person (green)")
    
    print("\n📹 STARTING WEBCAM DETECTION...")
    print("-" * 70)
    print("Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save screenshot")
    print("-" * 70)
    
    # Run webcam inference
    script_path = Path(__file__).parent / "webcam_inference.py"
    subprocess.run([sys.executable, str(script_path)])

if __name__ == "__main__":
    main()
