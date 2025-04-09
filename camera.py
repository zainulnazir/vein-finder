#!/usr/bin/env python3

import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2
from threading import Thread
from datetime import datetime

class VeinCamera:
    def __init__(self, resolution=(640, 480), framerate=30):
        """Initialize the camera with IR optimization for vein detection"""
        self.picam2 = Picamera2()
        
        # Configure camera for video mode using the working configuration
        config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        
        # Set camera parameters optimal for IR imaging
        self.picam2.set_controls({
            "ExposureTime": 20000,  # Higher exposure for IR
            "AnalogueGain": 6.0     # Gain setting from working example
        })
        
        self.resolution = resolution
        self.framerate = framerate
        self.frame = None
        self.stopped = False
        self.images_dir = "static/images"
        
        # Create directory for saved images if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
    def start(self):
        """Start the camera and capture thread"""
        self.picam2.start()
        # Wait for camera to initialize
        time.sleep(2)
        
        # Start the thread to read frames from the camera
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        """Update frame in background thread"""
        while True:
            if self.stopped:
                self.picam2.stop()
                return
                
            # Capture frame
            self.frame = self.picam2.capture_array()
    
    def read(self):
        """Return the current frame"""
        return self.frame
    
    def stop(self):
        """Stop the camera thread"""
        self.stopped = True
    
    def save_image(self, processed_frame=None):
        """Save the current frame or processed frame to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vein_{timestamp}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        
        if processed_frame is not None:
            cv2.imwrite(filepath, processed_frame)
        else:
            cv2.imwrite(filepath, self.frame)
            
        return filename
    
    def adjust_settings(self, exposure=None, gain=None):
        """Adjust camera settings for optimal vein imaging"""
        controls = {}
        
        if exposure is not None:
            controls["ExposureTime"] = int(exposure)
        
        if gain is not None:
            controls["AnalogueGain"] = float(gain)
            
        if controls:
            self.picam2.set_controls(controls)
    
    def get_camera_name(self):
        """Get camera model name"""
        try:
            return "Raspberry Pi Camera"
        except:
            return "Unknown Camera"
    
    @property
    def width(self):
        """Get the camera width"""
        return self.resolution[0]
    
    @property
    def height(self):
        """Get the camera height"""
        return self.resolution[1]
    
    @property
    def frame_rate(self):
        """Get the camera frame rate"""
        return self.framerate 