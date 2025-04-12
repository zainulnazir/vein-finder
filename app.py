#!/usr/bin/env python3

import os
import time
import sys
import argparse
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import json
from datetime import datetime
import threading
from skimage import filters, exposure  # Add missing imports for Frangi filter

# Import our custom modules
from camera import VeinCamera
from vein_detection import VeinDetector
from led_controller import LEDController

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Vein Finder - Advanced Medical Imaging System"
)
parser.add_argument(
    "--dev", action="store_true", help="Run in development mode with simulated hardware"
)
parser.add_argument(
    "--port", type=int, default=8000, help="Port to run the web server on"
)
parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
args = parser.parse_args()

app = Flask(__name__, static_folder="static", template_folder="templates")

# Directory for saving images
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "images"))
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Images will be saved to: {SAVE_DIR}")

# Global variables
camera = None
vein_detector = VeinDetector()
led_controller = None

# Settings
settings = {
    "detection_method": "adaptive",  # 'adaptive', 'frangi', 'laplacian'
    "contrast_method": "clahe",  # 'clahe', 'histogram_equalization', 'none'
    "led_brightness": 255,  # 0-255
    "led_pattern": 1,  # 1: all on, 2: alternate, 3: sequential
    "camera_exposure": 20000,  # Exposure time in microseconds
    "camera_gain": 6.0,  # Analog gain
    "zoom_level": 1.0,  # Zoom level: 1.0 is no zoom
    "rotation": 0,  # Rotation in degrees (0, 90, 180, 270)
    "stream_resolution": "720p",  # "480p", "720p", or "1080p"
}

# Image processing lock
processing_lock = threading.Lock()


def initialize_hardware():
    """Initialize camera and LED controller"""
    global camera, led_controller

    # Initialize camera
    if args.dev:
        print("Running in development mode: Simulated hardware")
        camera = VeinCamera(simulation_mode=True)
    else:
        camera = VeinCamera()

    # Set initial resolution from settings
    camera.set_resolution(settings["stream_resolution"])
    camera.start()

    # Initialize LED controller
    try:
        if args.dev:
            print("Using simulated LED controller")
            led_controller = LEDController(simulation=True)
        else:
            led_controller = LEDController()

        if led_controller.connected:
            led_controller.set_brightness(settings["led_brightness"])
            led_controller.set_pattern(settings["led_pattern"])
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
                # Apply zoom if needed
                current_zoom = settings.get("zoom_level", 1.0)
                if current_zoom > 1.0:
                    height, width = frame.shape[:2]

                    # Calculate aspect ratio of the frame
                    aspect_ratio = width / height

                    # Calculate crop dimensions based on zoom level while maintaining aspect ratio
                    crop_width = int(width / current_zoom)
                    crop_height = int(crop_width / aspect_ratio)

                    # If height calculation causes issues, recalculate to maintain aspect ratio
                    if crop_height > height:
                        crop_height = int(height / current_zoom)
                        crop_width = int(crop_height * aspect_ratio)

                    # Calculate center crop coordinates
                    start_x = int((width - crop_width) / 2)
                    start_y = int((height - crop_height) / 2)

                    # Crop the frame
                    frame = frame[
                        start_y : start_y + crop_height, start_x : start_x + crop_width
                    ]

                    # Resize back to original dimensions while maintaining aspect ratio
                    frame = cv2.resize(
                        frame, (width, height), interpolation=cv2.INTER_LINEAR
                    )

                # Apply rotation if needed
                current_rotation = settings.get("rotation", 0)
                if current_rotation > 0:
                    height, width = frame.shape[:2]
                    center = (width // 2, height // 2)

                    # Get rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(
                        center, current_rotation, 1.0
                    )

                    # Apply rotation
                    frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

                # Apply vein detection
                if settings["detection_method"] != "none":
                    processed_frame = vein_detector.detect_veins(
                        frame, method=settings["detection_method"]
                    )
                else:
                    # If no detection method is selected, just convert to RGB
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Encode frame as JPEG
                ret, buffer = cv2.imencode(".jpg", processed_frame)

                if not ret:
                    continue

                # Convert to bytes and yield
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)

        # Limit frame rate
        time.sleep(0.03)  # ~30 FPS


@app.route("/")
def index():
    """Render the main page"""
    print("Rendering index template")
    return render_template("index.html", settings=settings)


@app.route("/video_feed")
def video_feed():
    """Video streaming route"""
    print("Video feed requested")
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture", methods=["POST"])
def capture():
    """Capture and save current frame with enhanced vein visibility that matches the live stream"""
    global camera, vein_detector

    if camera is None:
        return jsonify({"success": False, "message": "Camera not initialized"})

    # Always capture at highest resolution (1080p)
    # Get multiple consecutive frames for better image quality
    frames = []
    for _ in range(5):  # Capture 5 frames for better averaging
        # Use high_res capture instead of regular read
        frame = camera.capture_high_res()
        if frame is not None:
            frames.append(frame)
        time.sleep(0.05)  # Short delay between captures

    if not frames:
        return jsonify({"success": False, "message": "Failed to capture frame"})

    with processing_lock:
        try:
            # IMPORTANT: Use exact same processing pipeline as live stream for consistency

            # 1. Save original frame at high quality
            original_frame = frames[0].copy()  # Keep one original frame

            # 2. Apply frame alignment and averaging for noise reduction
            aligned_frames = []
            # Use the first frame as reference
            reference = frames[0]

            # Convert reference to grayscale for alignment
            if len(reference.shape) > 2 and reference.shape[2] == 3:
                reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            else:
                reference_gray = reference.copy()

            # Add reference frame to aligned frames
            aligned_frames.append(reference)

            # Align and add other frames
            for i in range(1, len(frames)):
                frame = frames[i]

                # Convert to grayscale for alignment if needed
                if len(frame.shape) > 2 and frame.shape[2] == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame.copy()

                # Find transformation for alignment (ECC algorithm)
                try:
                    # Define transformation matrix
                    warp_mode = cv2.MOTION_TRANSLATION
                    warp_matrix = np.eye(2, 3, dtype=np.float32)

                    # Specify the number of iterations and threshold
                    criteria = (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.001,
                    )

                    # Run the ECC algorithm
                    _, warp_matrix = cv2.findTransformECC(
                        reference_gray, frame_gray, warp_matrix, warp_mode, criteria
                    )

                    # Apply the found transformation
                    aligned_frame = cv2.warpAffine(
                        frame,
                        warp_matrix,
                        (frame.shape[1], frame.shape[0]),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    )
                    aligned_frames.append(aligned_frame)
                except cv2.error:
                    # If alignment fails, use the original frame
                    aligned_frames.append(frame)

            # Calculate average of aligned frames for noise reduction
            aligned_avg = np.zeros_like(aligned_frames[0], dtype=np.float32)
            for aligned_frame in aligned_frames:
                aligned_avg += aligned_frame.astype(np.float32)
            aligned_avg /= len(aligned_frames)
            frame = aligned_avg.astype(np.uint8)

            # 3. Apply zoom and rotation if needed - SAME AS LIVE STREAM
            current_zoom = settings.get("zoom_level", 1.0)
            if current_zoom > 1.0:
                height, width = frame.shape[:2]

                # Calculate aspect ratio of the frame
                aspect_ratio = width / height

                # Calculate crop dimensions based on zoom level while maintaining aspect ratio
                crop_width = int(width / current_zoom)
                crop_height = int(crop_width / aspect_ratio)

                # If height calculation causes issues, recalculate to maintain aspect ratio
                if crop_height > height:
                    crop_height = int(height / current_zoom)
                    crop_width = int(crop_height * aspect_ratio)

                # Calculate center crop coordinates
                start_x = int((width - crop_width) / 2)
                start_y = int((height - crop_height) / 2)

                # Crop the frame
                frame = frame[
                    start_y : start_y + crop_height, start_x : start_x + crop_width
                ]

                # Resize back to original dimensions while maintaining aspect ratio
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_LINEAR
                )

            # Apply rotation if needed
            current_rotation = settings.get("rotation", 0)
            if current_rotation > 0:
                height, width = frame.shape[:2]
                center = (width // 2, height // 2)

                # Get rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, current_rotation, 1.0)

                # Apply rotation
                frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

            # 4. Apply the EXACT SAME vein detection method as in the live stream
            if settings["detection_method"] != "none":
                # The key is to use the same method that's used in the live stream
                processed_frame = vein_detector.detect_veins(
                    frame, method=settings["detection_method"]
                )
            else:
                # If no detection method is selected, just convert to RGB
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 5. Add metadata overlay with minimal impact on the image
            # Use smaller, less intrusive overlay than before
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Create a copy to avoid modifying the processed frame
            display_frame = processed_frame.copy()

            # Add small, semi-transparent overlay for metadata
            overlay = display_frame.copy()
            cv2.rectangle(
                overlay,
                (5, display_frame.shape[0] - 80),
                (250, display_frame.shape[0] - 5),
                (0, 0, 0),
                -1,
            )
            # Apply semi-transparency
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

            # Add text
            cv2.putText(
                display_frame,
                f"Patient: {request.json.get('patient_info', {}).get('name', 'Unknown')}",
                (10, display_frame.shape[0] - 55),
                font,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Time: {current_time}",
                (10, display_frame.shape[0] - 30),
                font,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Method: {settings['detection_method'].capitalize()}",
                (10, display_frame.shape[0] - 10),
                font,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Use display_frame for saving with metadata, but store processed_frame as the clean version
            clean_processed_frame = (
                processed_frame  # Store clean version without overlay
            )
            processed_frame = display_frame  # Use annotated version for saving

        except Exception as e:
            print(f"Error processing frame for capture: {e}")
            import traceback

            traceback.print_exc()

            # Fall back to original frame if processing fails
            if len(frame.shape) > 2 and frame.shape[2] == 3:
                processed_frame = frame.copy()  # Use original RGB frame
            else:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            original_frame = frame

    # Save both original and processed frames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_filename = f"original_{timestamp}.jpg"
    processed_filename = f"processed_{timestamp}.jpg"

    original_path = os.path.join(SAVE_DIR, original_filename)
    processed_path = os.path.join(SAVE_DIR, processed_filename)

    try:
        # Save with optimal quality
        cv2.imwrite(original_path, original_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(processed_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # Also save clean version (without overlay) if processing was successful
        if "clean_processed_frame" in locals():
            clean_filename = f"clean_processed_{timestamp}.jpg"
            clean_path = os.path.join(SAVE_DIR, clean_filename)
            cv2.imwrite(
                clean_path, clean_processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

        # Create detailed metadata
        metadata = {
            "timestamp": timestamp,
            "settings": settings.copy(),
            "original_image": original_filename,
            "processed_image": processed_filename,
            "clean_processed_image": (
                clean_filename if "clean_processed_frame" in locals() else None
            ),
            "camera_info": {
                "model": camera.get_camera_name(),
                "resolution": f"{camera.width}x{camera.height}",
                "exposure": settings["camera_exposure"],
                "gain": settings["camera_gain"],
            },
            "processing_info": {
                "method": settings["detection_method"],
                "contrast": settings["contrast_method"],
                "frames_averaged": len(aligned_frames),
            },
            "patient_info": request.json.get("patient_info", {}),
        }

        metadata_filename = f"metadata_{timestamp}.json"
        metadata_path = os.path.join(SAVE_DIR, metadata_filename)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return jsonify(
            {
                "success": True,
                "original_image": f"/static/images/{original_filename}",
                "processed_image": f"/static/images/{processed_filename}",
                "clean_processed_image": (
                    f"/static/images/{clean_filename}"
                    if "clean_processed_frame" in locals()
                    else None
                ),
                "timestamp": timestamp,
                "metadata": metadata,
            }
        )
    except Exception as e:
        print(f"Error saving images: {e}")
        return jsonify({"success": False, "message": f"Error saving images: {str(e)}"})


@app.route("/update_settings", methods=["POST"])
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
        if "camera_exposure" in data:
            camera.adjust_settings(exposure=settings["camera_exposure"])
        if "camera_gain" in data:
            camera.adjust_settings(gain=settings["camera_gain"])

    # Apply LED settings
    if led_controller is not None and led_controller.connected:
        if "led_brightness" in data:
            led_controller.set_brightness(settings["led_brightness"])
        if "led_pattern" in data:
            led_controller.set_pattern(settings["led_pattern"])

    return jsonify({"success": True, "settings": settings})


@app.route("/get_settings")
def get_settings():
    """Get current application settings"""
    global settings

    # Add LED status if available
    if led_controller is not None and led_controller.connected:
        try:
            # Try to get LED status (this is implementation dependent)
            settings["led_status"] = True  # Placeholder, should come from controller
        except:
            settings["led_status"] = False
    else:
        settings["led_status"] = False

    return jsonify({"success": True, "settings": settings})


@app.route("/camera_info")
def camera_info():
    """Get camera information"""
    global camera, settings

    if camera is None:
        camera_info = {
            "model": "Not connected",
            "resolution": "Unknown",
            "frame_rate": 0,
        }
    else:
        camera_info = {
            "model": camera.get_camera_name(),
            "resolution": f"{camera.width}x{camera.height}",
            "frame_rate": camera.frame_rate,
            "exposure": settings["camera_exposure"],
            "gain": settings["camera_gain"],
        }

    return jsonify({"success": True, "camera_info": camera_info})


@app.route("/images")
def list_images():
    """List all saved images"""
    images = []

    for filename in os.listdir(SAVE_DIR):
        if filename.startswith("processed_") and filename.endswith(".jpg"):
            timestamp = filename.replace("processed_", "").replace(".jpg", "")

            # Check for metadata
            metadata_file = os.path.join(SAVE_DIR, f"metadata_{timestamp}.json")
            metadata = None
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            images.append(
                {
                    "timestamp": timestamp,
                    "processed_image": f"/static/images/{filename}",
                    "original_image": f"/static/images/original_{timestamp}.jpg",
                    "metadata": metadata,
                }
            )

    # Sort by timestamp (newest first)
    images.sort(key=lambda x: x["timestamp"], reverse=True)

    return jsonify({"success": True, "images": images})


@app.route("/image_count")
def image_count():
    """Return the count of captured images"""
    count = 0
    for filename in os.listdir(SAVE_DIR):
        if filename.startswith("processed_") and filename.endswith(".jpg"):
            count += 1
    return jsonify({"success": True, "count": count})


@app.route("/clear_gallery", methods=["POST"])
def clear_gallery():
    """Clear all saved images"""
    try:
        # Only remove image files, not the directory itself
        for filename in os.listdir(SAVE_DIR):
            if filename.endswith(".jpg") or filename.endswith(".json"):
                file_path = os.path.join(SAVE_DIR, filename)
                os.remove(file_path)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error clearing gallery: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/save_patient", methods=["POST"])
def save_patient():
    """Save patient information"""
    try:
        data = request.get_json()

        # Create a patient data file
        if not os.path.exists("static/patient_data"):
            os.makedirs("static/patient_data")

        patient_id = data.get("patient_id", "unknown")

        with open(f"static/patient_data/{patient_id}.json", "w") as f:
            json.dump(data, f, indent=2)

        return jsonify({"success": True})
    except Exception as e:
        print(f"Error saving patient data: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/save_notes", methods=["POST"])
def save_notes():
    """Save procedure notes"""
    try:
        data = request.get_json()
        notes = data.get("notes", "")

        # Save notes to a file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not os.path.exists("static/notes"):
            os.makedirs("static/notes")

        with open(f"static/notes/notes_{timestamp}.txt", "w") as f:
            f.write(notes)

        return jsonify({"success": True})
    except Exception as e:
        print(f"Error saving notes: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/zoom", methods=["POST"])
def zoom():
    """Handle zoom in/out requests"""
    global settings

    try:
        data = request.get_json()
        action = data.get("action", "in")

        # Update zoom level in settings
        current_zoom = settings.get("zoom_level", 1.0)

        if action == "in":
            settings["zoom_level"] = min(current_zoom + 0.2, 3.0)  # Max zoom 3x
        else:
            settings["zoom_level"] = max(current_zoom - 0.2, 1.0)  # Min zoom 1x

        return jsonify({"success": True, "zoom_level": settings["zoom_level"]})
    except Exception as e:
        print(f"Error handling zoom: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/rotate", methods=["POST"])
def rotate():
    """Handle rotation requests"""
    global settings

    try:
        # Toggle rotation in 90-degree increments
        current_rotation = settings.get("rotation", 0)
        settings["rotation"] = (current_rotation + 90) % 360

        return jsonify({"success": True, "rotation": settings["rotation"]})
    except Exception as e:
        print(f"Error handling rotation: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/delete_image", methods=["POST"])
def delete_image():
    """Delete a specific image"""
    try:
        data = request.get_json()
        image_path = data.get("image_path", "")

        print(f"Attempting to delete image: {image_path}")

        # Extract just the filename
        filename = os.path.basename(image_path)

        # Handle case where full URL path is provided
        if "/static/images/" in image_path:
            filename = image_path.split("/static/images/")[1]

        # Get timestamp part for associated files
        if filename.startswith("processed_"):
            timestamp = filename.replace("processed_", "").replace(".jpg", "")

            # Delete all associated files
            files_to_delete = [
                os.path.join(SAVE_DIR, f"processed_{timestamp}.jpg"),
                os.path.join(SAVE_DIR, f"original_{timestamp}.jpg"),
                os.path.join(SAVE_DIR, f"metadata_{timestamp}.json"),
            ]

            deleted_count = 0
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted: {file_path}")
                else:
                    print(f"File not found: {file_path}")

            print(
                f"Deleted {deleted_count} files associated with timestamp {timestamp}"
            )
            return jsonify({"success": True, "deleted_count": deleted_count})
        else:
            print(f"Invalid filename format: {filename}")
            return jsonify(
                {"success": False, "message": f"Invalid image path format: {filename}"}
            )
    except Exception as e:
        print(f"Error deleting image: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/toggle_leds", methods=["POST"])
def toggle_leds():
    """Toggle LED lights on/off"""
    global led_controller

    if led_controller is None or not led_controller.connected:
        return jsonify({"success": False, "message": "LED controller not connected"})

    try:
        # Toggle LED state
        led_status = led_controller.toggle_leds()
        return jsonify({"success": True, "led_status": led_status})
    except Exception as e:
        print(f"Error toggling LEDs: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shutdown camera and LED controller properly, then shut down the Raspberry Pi"""
    global camera, led_controller

    try:
        # First shutdown all peripherals
        if camera is not None:
            camera.stop()
            print("Camera stopped.")

        if led_controller is not None and led_controller.connected:
            # Turn off LEDs before disconnecting
            led_controller.toggle_leds(False)
            print("LEDs turned off.")

        # Start a background thread to shut down the system
        # This allows the response to be sent before system shuts down
        def system_shutdown():
            time.sleep(2)  # Wait for response to be sent
            print("Initiating system shutdown...")

            # Skip on macOS (development)
            if sys.platform == "darwin":
                print("Running on macOS, shutdown command skipped.")
                return

            # On Raspberry Pi: Try different shutdown methods
            try:
                # Method 1: Direct password input using echo to sudo
                print("Attempting shutdown with password input...")
                password = "Zainbuttzb123"  # Store the password in a variable
                shutdown_cmd = f"echo '{password}' | sudo -S shutdown -h now"
                result = os.system(shutdown_cmd)
                print(f"Shutdown command result: {result}")
                if result == 0:
                    print("Shutdown command executed successfully")
                    return
                else:
                    print("Shutdown command failed with direct password input")
            except Exception as e:
                print(f"Error during shutdown with password: {e}")

            try:
                # Method 2: Write password to a temporary file and use sudo -S with redirection
                print("Attempting alternate shutdown method...")
                with open("/tmp/pwd.txt", "w") as f:
                    f.write(f"Zainbuttzb123\n")
                os.chmod("/tmp/pwd.txt", 0o600)  # Secure the file
                os.system("cat /tmp/pwd.txt | sudo -S shutdown -h now")
                # Clean up the password file
                os.remove("/tmp/pwd.txt")
                return
            except Exception as e:
                print(f"Alternate shutdown method failed: {e}")

            try:
                # Method 3: Try using sudo with systemctl
                print("Attempting system shutdown with systemctl...")
                os.system('echo "Zainbuttzb123" | sudo -S systemctl poweroff')
                return
            except Exception as e:
                print(f"Systemctl poweroff failed: {e}")

            try:
                # Method 4: Try halt command as a last resort
                print("Attempting shutdown with halt command...")
                os.system('echo "Zainbuttzb123" | sudo -S halt -p')
                return
            except Exception as e:
                print(f"Halt command failed: {e}")

            print("All shutdown methods failed. Please shut down manually.")

        thread = threading.Thread(target=system_shutdown)
        thread.daemon = True
        thread.start()

        print("Shutdown process initiated.")
        return jsonify({"success": True, "message": "System is shutting down..."})
    except Exception as e:
        print(f"Error during shutdown: {e}")
        return jsonify({"success": False, "message": f"Shutdown error: {str(e)}"})


@app.route("/hardware_status")
def hardware_status():
    """Return the current status of connected hardware"""
    global camera, led_controller

    # Check camera status
    camera_status = {
        "connected": camera is not None,
        "simulation": camera is not None
        and hasattr(camera, "simulation")
        and camera.simulation,
        "name": camera.get_camera_name() if camera is not None else "Not Connected",
    }

    # Check LED controller status
    led_status = {
        "connected": led_controller is not None and led_controller.connected,
        "port": (
            getattr(led_controller.ser, "port", "Not Connected")
            if led_controller is not None and led_controller.connected
            else "Not Connected"
        ),
    }

    return jsonify(
        {"success": True, "camera": camera_status, "led_controller": led_status}
    )


@app.route("/change_resolution", methods=["POST"])
def change_resolution():
    """Change the streaming resolution of the camera"""
    global camera, settings

    if camera is None:
        return jsonify({"success": False, "message": "Camera not initialized"})

    try:
        data = request.get_json()
        resolution = data.get("resolution", "720p")

        # Validate resolution option
        if resolution not in ["480p", "720p", "1080p"]:
            return jsonify({"success": False, "message": "Invalid resolution"})

        # Set the new resolution
        success = camera.set_resolution(resolution)

        if success:
            # Update settings
            settings["stream_resolution"] = resolution

            return jsonify(
                {
                    "success": True,
                    "message": f"Resolution changed to {resolution}",
                    "resolution": resolution,
                    "width": camera.width,
                    "height": camera.height,
                }
            )
        else:
            return jsonify({"success": False, "message": "Failed to change resolution"})
    except Exception as e:
        print(f"Error changing resolution: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/test")
def test_page():
    """Simple test route to check if Flask is working"""
    return "Flask server is running! Try accessing the main page at '/'."


if __name__ == "__main__":
    # Initialize hardware
    initialize_hardware()

    try:
        # Start Flask app
        app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=True)
    finally:
        # Clean up resources
        if camera is not None:
            camera.stop()

        if led_controller is not None and led_controller.connected:
            led_controller.toggle_leds(False)
