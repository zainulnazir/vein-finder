#!/usr/bin/env python3

import os
import time
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import json
from datetime import datetime
import threading

# Import our custom modules
from camera import VeinCamera
from vein_detection import VeinDetector
from led_controller import LEDController

app = Flask(__name__, static_folder='static', template_folder='templates')

# Directory for saving images
SAVE_DIR = os.path.join('static', 'images')
os.makedirs(SAVE_DIR, exist_ok=True)

# Global variables
camera = None
vein_detector = VeinDetector()
led_controller = None

# Settings
settings = {
    'detection_method': 'adaptive',  # 'adaptive', 'frangi', 'laplacian'
    'contrast_method': 'clahe',      # 'clahe', 'histogram_equalization', 'none'
    'led_brightness': 255,           # 0-255
    'led_pattern': 1,                # 1: all on, 2: alternate, 3: sequential
    'camera_exposure': 20000,        # Exposure time in microseconds
    'camera_gain': 6.0,              # Analog gain
    'zoom_level': 1.0,               # Zoom level: 1.0 is no zoom
    'rotation': 0                    # Rotation in degrees (0, 90, 180, 270)
}

# Image processing lock
processing_lock = threading.Lock()

def initialize_hardware():
    """Initialize camera and LED controller"""
    global camera, led_controller
    
    # Initialize camera
    camera = VeinCamera()
    camera.start()
    
    # Initialize LED controller
    try:
        led_controller = LEDController()
        if led_controller.connected:
            led_controller.set_brightness(settings['led_brightness'])
            led_controller.set_pattern(settings['led_pattern'])
    except Exception as e:
        print(f"Error initializing LED controller: {e}")
        led_controller = None

def generate_frames():
    """Generate frames for video streaming"""
    global camera, vein_detector
    
    if camera is None:
        return
        
    while True:
        # Get frame from camera
        frame = camera.read()
        if frame is None:
            time.sleep(0.1)  # Wait a bit and try again
            continue
            
        # Process frame for vein detection
        with processing_lock:
            try:
                if settings['detection_method'] != 'none':
                    processed_frame = vein_detector.detect_veins(frame, method=settings['detection_method'])
                else:
                    # If no detection method is selected, just convert to RGB
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                
                if not ret:
                    continue
                    
                # Convert to bytes and yield
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)
               
        # Limit frame rate
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    """Render the main page"""
    print("Rendering index template")
    return render_template('index.html', settings=settings)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    print("Video feed requested")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """Capture and save current frame"""
    global camera, vein_detector
    
    if camera is None:
        return jsonify({'success': False, 'message': 'Camera not initialized'})
    
    # Get current frame
    frame = camera.read()
    if frame is None:
        return jsonify({'success': False, 'message': 'Failed to capture frame'})
    
    # Process frame if needed
    with processing_lock:
        if settings['detection_method'] != 'none':
            try:
                processed_frame = vein_detector.detect_veins(frame, method=settings['detection_method'])
            except Exception as e:
                print(f"Error processing frame for capture: {e}")
                processed_frame = frame.copy()
        else:
            processed_frame = frame.copy()
    
    # Save both original and processed frames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    original_filename = f"original_{timestamp}.jpg"
    processed_filename = f"processed_{timestamp}.jpg"
    
    original_path = os.path.join(SAVE_DIR, original_filename)
    processed_path = os.path.join(SAVE_DIR, processed_filename)
    
    try:
        cv2.imwrite(original_path, frame)
        cv2.imwrite(processed_path, processed_frame)
        
        # Create metadata
        metadata = {
            'timestamp': timestamp,
            'settings': settings.copy(),
            'original_image': original_filename,
            'processed_image': processed_filename
        }
        
        metadata_filename = f"metadata_{timestamp}.json"
        metadata_path = os.path.join(SAVE_DIR, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'success': True,
            'original_image': f"/static/images/{original_filename}",
            'processed_image': f"/static/images/{processed_filename}",
            'metadata': metadata
        })
    except Exception as e:
        print(f"Error saving images: {e}")
        return jsonify({'success': False, 'message': f'Error saving images: {str(e)}'})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    global settings, camera, led_controller
    
    data = request.get_json()
    
    # Update settings
    for key, value in data.items():
        if key in settings:
            settings[key] = value
    
    # Apply camera settings
    if camera is not None:
        if 'camera_exposure' in data:
            camera.adjust_settings(exposure=settings['camera_exposure'])
        if 'camera_gain' in data:
            camera.adjust_settings(gain=settings['camera_gain'])
    
    # Apply LED settings
    if led_controller is not None and led_controller.connected:
        if 'led_brightness' in data:
            led_controller.set_brightness(settings['led_brightness'])
        if 'led_pattern' in data:
            led_controller.set_pattern(settings['led_pattern'])
    
    return jsonify({'success': True, 'settings': settings})

@app.route('/get_settings')
def get_settings():
    """Get current application settings"""
    global settings
    
    # Add LED status if available
    if led_controller is not None and led_controller.connected:
        try:
            # Try to get LED status (this is implementation dependent)
            settings['led_status'] = True  # Placeholder, should come from controller
        except:
            settings['led_status'] = False
    else:
        settings['led_status'] = False
    
    return jsonify({'success': True, 'settings': settings})

@app.route('/camera_info')
def camera_info():
    """Get camera information"""
    global camera, settings
    
    if camera is None:
        camera_info = {
            'model': 'Not connected',
            'resolution': 'Unknown',
            'frame_rate': 0
        }
    else:
        camera_info = {
            'model': camera.get_camera_name(),
            'resolution': f"{camera.width}x{camera.height}",
            'frame_rate': camera.frame_rate,
            'exposure': settings['camera_exposure'],
            'gain': settings['camera_gain']
        }
    
    return jsonify({
        'success': True,
        'camera_info': camera_info
    })

@app.route('/images')
def list_images():
    """List all saved images"""
    images = []
    
    for filename in os.listdir(SAVE_DIR):
        if filename.startswith('processed_') and filename.endswith('.jpg'):
            timestamp = filename.replace('processed_', '').replace('.jpg', '')
            
            # Check for metadata
            metadata_file = os.path.join(SAVE_DIR, f"metadata_{timestamp}.json")
            metadata = None
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            images.append({
                'timestamp': timestamp,
                'processed_image': f"/static/images/{filename}",
                'original_image': f"/static/images/original_{timestamp}.jpg",
                'metadata': metadata
            })
    
    # Sort by timestamp (newest first)
    images.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({'success': True, 'images': images})

@app.route('/image_count')
def image_count():
    """Return the count of captured images"""
    count = 0
    for filename in os.listdir(SAVE_DIR):
        if filename.startswith('processed_') and filename.endswith('.jpg'):
            count += 1
    return jsonify({'success': True, 'count': count})

@app.route('/clear_gallery', methods=['POST'])
def clear_gallery():
    """Clear all saved images"""
    try:
        # Only remove image files, not the directory itself
        for filename in os.listdir(SAVE_DIR):
            if filename.endswith('.jpg') or filename.endswith('.json'):
                file_path = os.path.join(SAVE_DIR, filename)
                os.remove(file_path)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error clearing gallery: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/save_patient', methods=['POST'])
def save_patient():
    """Save patient information"""
    try:
        data = request.get_json()
        
        # Create a patient data file
        if not os.path.exists('static/patient_data'):
            os.makedirs('static/patient_data')
        
        patient_id = data.get('patient_id', 'unknown')
        
        with open(f'static/patient_data/{patient_id}.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving patient data: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/save_notes', methods=['POST'])
def save_notes():
    """Save procedure notes"""
    try:
        data = request.get_json()
        notes = data.get('notes', '')
        
        # Save notes to a file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not os.path.exists('static/notes'):
            os.makedirs('static/notes')
        
        with open(f'static/notes/notes_{timestamp}.txt', 'w') as f:
            f.write(notes)
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving notes: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/zoom', methods=['POST'])
def zoom():
    """Handle zoom in/out requests"""
    global settings
    
    try:
        data = request.get_json()
        action = data.get('action', 'in')
        
        # Update zoom level in settings
        current_zoom = settings.get('zoom_level', 1.0)
        
        if action == 'in':
            settings['zoom_level'] = min(current_zoom + 0.2, 3.0)  # Max zoom 3x
        else:
            settings['zoom_level'] = max(current_zoom - 0.2, 1.0)  # Min zoom 1x
        
        return jsonify({'success': True, 'zoom_level': settings['zoom_level']})
    except Exception as e:
        print(f"Error handling zoom: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/rotate', methods=['POST'])
def rotate():
    """Handle rotation requests"""
    global settings
    
    try:
        # Toggle rotation in 90-degree increments
        current_rotation = settings.get('rotation', 0)
        settings['rotation'] = (current_rotation + 90) % 360
        
        return jsonify({'success': True, 'rotation': settings['rotation']})
    except Exception as e:
        print(f"Error handling rotation: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/delete_image', methods=['POST'])
def delete_image():
    """Delete a specific image"""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        
        # Extract just the filename
        filename = os.path.basename(image_path)
        
        # Get timestamp part for associated files
        if filename.startswith('processed_'):
            timestamp = filename.replace('processed_', '').replace('.jpg', '')
            
            # Delete all associated files
            files_to_delete = [
                os.path.join(SAVE_DIR, f"processed_{timestamp}.jpg"),
                os.path.join(SAVE_DIR, f"original_{timestamp}.jpg"),
                os.path.join(SAVE_DIR, f"metadata_{timestamp}.json")
            ]
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid image path format'})
    except Exception as e:
        print(f"Error deleting image: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/toggle_leds', methods=['POST'])
def toggle_leds():
    """Toggle LED lights on/off"""
    global led_controller
    
    if led_controller is None or not led_controller.connected:
        return jsonify({'success': False, 'message': 'LED controller not connected'})
    
    try:
        # Toggle LED state
        led_status = led_controller.toggle_leds()
        return jsonify({'success': True, 'led_status': led_status})
    except Exception as e:
        print(f"Error toggling LEDs: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown camera and LED controller properly"""
    global camera, led_controller
    
    if camera is not None:
        camera.stop()
    
    if led_controller is not None and led_controller.connected:
        # Turn off LEDs before disconnecting
        led_controller.toggle_leds(False)
    
    return jsonify({'success': True})

@app.route('/test')
def test_page():
    """Simple test route to check if Flask is working"""
    return "Flask server is running! Try accessing the main page at '/'."

if __name__ == '__main__':
    # Initialize hardware
    initialize_hardware()
    
    try:
        # Start Flask app
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    finally:
        # Clean up resources
        if camera is not None:
            camera.stop()
        
        if led_controller is not None and led_controller.connected:
            led_controller.toggle_leds(False) 