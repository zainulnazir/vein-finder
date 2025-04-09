#!/usr/bin/env python3

"""
Simple test script for the Raspberry Pi camera
Use this to verify that the camera is working properly
"""

from picamera2 import Picamera2
import time
import cv2
import os

def test_camera():
    print("Initializing camera...")
    
    # Initialize camera with video configuration
    camera = Picamera2()
    config = camera.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    camera.configure(config)
    
    # Set camera parameters
    camera.set_controls({
        "ExposureTime": 20000,
        "AnalogueGain": 6.0
    })
    
    print("Starting camera...")
    camera.start()
    time.sleep(2)  # Camera warm-up
    
    print("Capturing test image...")
    frame = camera.capture_array()
    
    # Ensure output directory exists
    os.makedirs("test_output", exist_ok=True)
    
    # Save raw image
    cv2.imwrite("test_output/raw_test.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Save grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("test_output/gray_test.jpg", gray)
    
    # Apply some processing similar to vein detection
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imwrite("test_output/enhanced_test.jpg", enhanced)
    
    print("Camera test complete. Images saved to test_output/")
    
    # Clean up
    camera.stop()
    return True

if __name__ == "__main__":
    try:
        success = test_camera()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Camera test failed: {e}")
        exit(1) 