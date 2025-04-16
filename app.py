#!/usr/bin/env python3

import os
import time
import sys
import argparse
from flask import Flask, render_template, Response, request, jsonify, url_for
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

# Create directory for position guide images
GUIDE_DIR = os.path.abspath(os.path.join(SAVE_DIR, "guides"))
os.makedirs(GUIDE_DIR, exist_ok=True)

# Create directory for patient data and notes
PATIENT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "static", "patient_data")
)
os.makedirs(PATIENT_DATA_DIR, exist_ok=True)
NOTES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "notes"))
os.makedirs(NOTES_DIR, exist_ok=True)

# Global variables
camera = None
vein_detector = VeinDetector()
led_controller = None

# Settings - Add new tunable parameters
settings = {
    "detection_method": "adaptive",  # 'adaptive', 'frangi', 'laplacian', 'thermal'
    # Contrast moved to VeinDetector params
    "led_brightness": 255,
    "led_pattern": 1,
    "camera_exposure": 20000,
    "camera_gain": 6.0,
    "zoom_level": 1.0,
    "rotation": 0,
    "stream_resolution": "480p",
    # CLAHE Params (Matches VeinDetector defaults)
    "clahe_clip_limit": 5.0,
    "clahe_tile_grid_size": 8,
    # Frangi Params (Matches VeinDetector defaults)
    "frangi_scale_min": 1.0,
    "frangi_scale_max": 8.0,
    "frangi_scale_step": 1.0,
    "frangi_beta": 0.5,
    "frangi_gamma": 15,
}

# Image processing lock
processing_lock = threading.Lock()

# Initialize VeinDetector with current settings
vein_detector.update_params(settings)


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
                # Only apply zoom if zoom level is greater than 1.0
                if current_zoom > 1.0:
                    height, width = frame.shape[:2]

                    # Ensure frame has valid dimensions
                    if height > 0 and width > 0:
                        # Calculate aspect ratio of the frame
                        aspect_ratio = width / height

                        # Calculate crop dimensions based on zoom level while maintaining aspect ratio
                        crop_width = int(width / current_zoom)
                        crop_height = int(
                            height / current_zoom
                        )  # Use height for primary calc to avoid potential aspect ratio issues

                        # Ensure crop dimensions are valid
                        if crop_width <= 0:
                            crop_width = 1
                        if crop_height <= 0:
                            crop_height = 1

                        # Recalculate the secondary dimension to maintain aspect ratio strictly
                        # If we base on width:
                        # calc_height = int(crop_width / aspect_ratio)
                        # If we base on height:
                        # calc_width = int(crop_height * aspect_ratio)
                        # Let's base on height as it's often the limiting factor in vertical video space
                        crop_width = int(crop_height * aspect_ratio)
                        if (
                            crop_width > width
                        ):  # Recalculate if calculated width exceeds original
                            crop_width = width
                            crop_height = int(crop_width / aspect_ratio)

                        # Calculate center crop coordinates
                        start_x = max(0, int((width - crop_width) / 2))
                        start_y = max(0, int((height - crop_height) / 2))

                        # Ensure coordinates + dimensions do not exceed frame boundaries
                        end_x = min(width, start_x + crop_width)
                        end_y = min(height, start_y + crop_height)
                        crop_width = end_x - start_x
                        crop_height = end_y - start_y

                        # Only proceed if crop dimensions are valid
                        if crop_width > 0 and crop_height > 0:
                            # Crop the frame
                            frame_cropped = frame[start_y:end_y, start_x:end_x]

                            # Resize back to original dimensions
                            # Using INTER_LINEAR is generally good for downscaling/upscaling
                            frame = cv2.resize(
                                frame_cropped,
                                (width, height),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        else:
                            print(
                                f"Warning: Invalid crop dimensions calculated for zoom {current_zoom}. Skipping zoom."
                            )
                    else:
                        print("Warning: Frame has zero dimension, skipping zoom.")

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
    """Capture and save a single high-quality frame with enhanced vein visibility."""
    global camera, vein_detector

    if camera is None:
        return jsonify({"success": False, "message": "Camera not initialized"})

    # Get detection method from request or use current settings
    data = request.get_json()
    detection_method = data.get("detection_method", settings["detection_method"])

    # Capture ONE high-resolution frame
    print(f"Capturing single high-resolution frame with method: {detection_method}...")
    frame = camera.capture_high_res()

    if frame is None:
        print("Error: Failed to capture high-resolution frame.")
        return jsonify({"success": False, "message": "Failed to capture frame"})
    print(f"Captured frame shape: {frame.shape}")

    with processing_lock:
        try:
            # Keep a copy of the original high-res frame before processing
            original_frame = frame.copy()

            # Apply zoom and rotation if needed
            current_zoom = settings.get("zoom_level", 1.0)
            if current_zoom > 1.0:
                height, width = frame.shape[:2]
                if height > 0 and width > 0:
                    aspect_ratio = width / height
                    crop_width = int(width / current_zoom)
                    crop_height = int(height / current_zoom)
                    if crop_width <= 0:
                        crop_width = 1
                    if crop_height <= 0:
                        crop_height = 1
                    crop_width = int(crop_height * aspect_ratio)
                    if crop_width > width:
                        crop_width = width
                        crop_height = int(crop_width / aspect_ratio)
                    start_x = max(0, int((width - crop_width) / 2))
                    start_y = max(0, int((height - crop_height) / 2))
                    end_x = min(width, start_x + crop_width)
                    end_y = min(height, start_y + crop_height)
                    crop_width = end_x - start_x
                    crop_height = end_y - start_y
                    if crop_width > 0 and crop_height > 0:
                        frame_cropped = frame[start_y:end_y, start_x:end_x]
                        frame = cv2.resize(
                            frame_cropped,
                            (width, height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        print(
                            f"Warning: Invalid crop dimensions for zoom {current_zoom}. Skipping zoom."
                        )
                else:
                    print("Warning: Frame has zero dimension, skipping zoom.")

            # Apply rotation if needed
            current_rotation = settings.get("rotation", 0)
            if current_rotation > 0:
                height, width = frame.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, current_rotation, 1.0)
                frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

            # Apply vein detection using the specified method
            if detection_method != "none":
                print(f"Applying vein detection method: {detection_method}")
                processed_frame = vein_detector.detect_veins(
                    frame, method=detection_method
                )
            else:
                print("No vein detection selected, using original frame.")
                # If no detection method is selected, just use the (potentially zoomed/rotated) frame
                # Ensure it's in BGR format for saving if needed
                if len(frame.shape) == 2:  # Grayscale
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif (
                    len(frame.shape) == 3 and frame.shape[2] == 1
                ):  # Grayscale with channel dim
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 3:  # Already BGR/RGB
                    processed_frame = frame  # Assume BGR or compatible
                else:
                    print(
                        f"Warning: Unexpected frame format {frame.shape}. Using as is."
                    )
                    processed_frame = frame  # Fallback

            # Add metadata overlay - Use the detection method from the request
            if "patient_info" in data:
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
                font = cv2.FONT_HERSHEY_SIMPLEX
                patient_name = data.get("patient_info", {}).get("name", "Unknown")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                cv2.putText(
                    display_frame,
                    f"Patient: {patient_name}",
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
                    f"Method: {detection_method.capitalize()}",
                    (10, display_frame.shape[0] - 10),
                    font,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                processed_frame = display_frame

            # Save the processed frame and the original frame
            processed_filename = camera.save_image(processed_frame)
            original_filename = camera.save_image(original_frame, suffix="_original")

            if processed_filename is None:
                raise ValueError("Failed to save processed image")
            if original_filename is None:
                print(
                    "Warning: Failed to save original image, but proceeding."
                )  # Don't fail if original save fails

            print(f"Processed image saved as: {processed_filename}")
            if original_filename:
                print(f"Original image saved as: {original_filename}")

            # Extract timestamp from filename
            timestamp_str = "unknown"
            try:
                parts = processed_filename.replace("vein_", "").split(".")[0].split("_")
                if len(parts) == 3:
                    # Format: YYYYMMDD_HHMMSS_ffffff -> Convert to YYYYMMDD_HHMMSS for JS
                    timestamp_str = f"{parts[0]}_{parts[1]}"
                elif len(parts) == 2:  # Legacy format? YYYYMMDD_HHMMSS
                    timestamp_str = f"{parts[0]}_{parts[1]}"
            except Exception as e:
                print(
                    f"Error parsing timestamp from filename '{processed_filename}': {e}"
                )

            # Save metadata including patient info and detection method
            metadata = {
                "timestamp": timestamp_str,
                "detection_method": detection_method,
                "patient_info": data.get("patient_info", {}),
                "camera_settings": {
                    "exposure": settings["camera_exposure"],
                    "gain": settings["camera_gain"],
                    "zoom": settings["zoom_level"],
                    "rotation": settings["rotation"],
                },
            }

            # Save metadata to a file
            metadata_filename = f"metadata_{timestamp_str}.json"
            with open(os.path.join(SAVE_DIR, metadata_filename), "w") as f:
                json.dump(metadata, f, indent=2)

            # Return success with filenames AND timestamp in the format JS expects
            processed_url = url_for("static", filename=f"images/{processed_filename}")
            original_url = (
                url_for("static", filename=f"images/{original_filename}")
                if original_filename
                else None
            )

            return jsonify(
                {
                    "success": True,
                    "message": "Image captured successfully",
                    "processed_image": processed_url,
                    "original_image": original_url,
                    "timestamp": timestamp_str,
                    "detection_method": detection_method,
                }
            )

        except Exception as e:
            print(f"Error processing captured frame: {e}")
            return jsonify({"success": False, "message": f"Error processing: {e}"})


@app.route("/update_settings", methods=["POST"])
def update_settings():
    """Update settings for camera, LEDs, and detection parameters"""
    global settings, camera, led_controller, vein_detector

    data = request.get_json()
    print(f"Received settings update request: {data}")

    updated = False
    requires_camera_reconfig = False

    # Update individual settings if provided
    for key, value in data.items():
        if key in settings:
            # Type conversion based on expected type
            current_type = type(settings[key])
            try:
                new_value = current_type(value)
                if settings[key] != new_value:
                    settings[key] = new_value
                    print(f"Updated setting: {key} = {new_value}")
                    updated = True
                    # Check if camera needs reconfiguring
                    if key == "stream_resolution":
                        requires_camera_reconfig = True
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Could not convert value '{value}' for key '{key}': {e}"
                )

    # Update hardware if settings changed
    if updated:
        # Camera settings (Exposure, Gain)
        if "camera_exposure" in data or "camera_gain" in data:
            if camera and not camera.simulation:
                camera.adjust_settings(
                    exposure=settings["camera_exposure"],
                    gain=settings["camera_gain"],
                )

        # Camera resolution
        if requires_camera_reconfig:
            if camera and not camera.simulation:
                success = camera.set_resolution(settings["stream_resolution"])
                if not success:
                    # Revert setting if change failed
                    # Need to fetch previous state or handle error better
                    print(
                        f"ERROR: Failed to set resolution to {settings['stream_resolution']}"
                    )
                    return jsonify(
                        {"success": False, "message": "Failed to set camera resolution"}
                    )

        # LED settings
        if "led_brightness" in data or "led_pattern" in data:
            if led_controller and led_controller.connected:
                if "led_brightness" in data:
                    led_controller.set_brightness(settings["led_brightness"])
                if "led_pattern" in data:
                    led_controller.set_pattern(settings["led_pattern"])

        # Vein Detector Parameters (CLAHE, Frangi)
        # Collect relevant params and update the detector
        detector_params = {
            k: v
            for k, v in settings.items()
            if k
            in [
                "clahe_clip_limit",
                "clahe_tile_grid_size",
                "frangi_scale_min",
                "frangi_scale_max",
                "frangi_scale_step",
                "frangi_beta",
                "frangi_gamma",
            ]
        }
        if (
            any(key in data for key in detector_params) or "detection_method" in data
        ):  # Update if relevant keys were in request
            vein_detector.update_params(detector_params)
            print(f"Updated VeinDetector params: {detector_params}")

        return jsonify({"success": True, "settings": settings})
    else:
        # No change in settings relevant to this request
        return jsonify({"success": True, "message": "No relevant settings changed"})


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
            "simulation": hasattr(camera, "simulation") and camera.simulation,
        }

    return jsonify({"success": True, "camera_info": camera_info})


@app.route("/images")
def list_images():
    """List all saved images based on the 'vein_' prefix with pagination support."""
    images = []
    processed_files = {}

    # First pass: Find all 'vein_' prefixed files
    for filename in os.listdir(SAVE_DIR):
        if filename.startswith("vein_") and filename.endswith(".jpg"):
            # Extract potential timestamp (YYYYMMDD_HHMMSS_ffffff)
            try:
                base_name = filename.replace("vein_", "").replace(".jpg", "")
                is_original = base_name.endswith("_original")
                timestamp = base_name.replace("_original", "")

                if timestamp not in processed_files:
                    processed_files[timestamp] = {"processed": None, "original": None}

                if is_original:
                    processed_files[timestamp]["original"] = filename
                else:
                    processed_files[timestamp]["processed"] = filename
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue

    # Second pass: Construct the image list
    for timestamp, files in processed_files.items():
        if files["processed"]:
            processed_url = url_for("static", filename=f'images/{files["processed"]}')
            original_url = None
            if files["original"]:
                original_url = url_for("static", filename=f'images/{files["original"]}')

            # Extract cleaner timestamp for display if possible (YYYYMMDD_HHMMSS)
            display_timestamp = (
                timestamp.split("_")[0] + "_" + timestamp.split("_")[1]
                if len(timestamp.split("_")) >= 2
                else timestamp
            )

            # Try to load metadata if available
            metadata = None
            metadata_path = os.path.join(SAVE_DIR, f"metadata_{display_timestamp}.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"Error loading metadata for {display_timestamp}: {e}")

            images.append(
                {
                    "timestamp": display_timestamp,
                    "processed_image": processed_url,
                    "original_image": original_url,
                    "metadata": metadata,
                }
            )

    # Sort by the timestamp (newest first)
    images.sort(key=lambda x: x["timestamp"] if x["timestamp"] else "", reverse=True)

    print(f"Found {len(images)} image pairs.")
    return jsonify({"success": True, "images": images})


@app.route("/image_count")
def image_count():
    """Return the count of captured images and the timestamp of the last capture"""
    processed_images = []
    last_capture = None

    for filename in os.listdir(SAVE_DIR):
        if (
            filename.startswith("vein_")
            and filename.endswith(".jpg")
            and not filename.endswith("_original.jpg")
        ):
            processed_images.append(filename)

    count = len(processed_images)

    # Find the timestamp of the last capture
    if processed_images:
        try:
            # Sort by modification time, newest first
            processed_images.sort(
                key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True
            )
            newest_filename = processed_images[0]

            # Extract timestamp from filename
            timestamp = newest_filename.replace("vein_", "").replace(".jpg", "")
            if "_" in timestamp:
                last_capture = (
                    timestamp.split("_")[0] + "_" + timestamp.split("_")[1]
                    if len(timestamp.split("_")) >= 2
                    else timestamp
                )
            else:
                last_capture = timestamp
        except Exception as e:
            print(f"Error determining last capture: {e}")

    return jsonify({"success": True, "count": count, "last_capture": last_capture})


@app.route("/clear_gallery", methods=["POST"])
def clear_gallery():
    """Clear all saved images"""
    try:
        # Only remove image files, not the directory itself
        for filename in os.listdir(SAVE_DIR):
            if filename.endswith(".jpg") or filename.endswith(".json"):
                file_path = os.path.join(SAVE_DIR, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
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
        patient_id = data.get("patient_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{patient_id}_{timestamp}.json"
            if patient_id != "unknown"
            else f"patient_{timestamp}.json"
        )

        with open(os.path.join(PATIENT_DATA_DIR, filename), "w") as f:
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
        filename = f"notes_{timestamp}.txt"

        with open(os.path.join(NOTES_DIR, filename), "w") as f:
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
    """Delete a specific image and all associated files"""
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
        if filename.startswith("vein_"):
            timestamp = filename.replace("vein_", "").replace(".jpg", "")

            # Extract the timestamp without microseconds
            if "_" in timestamp:
                main_timestamp = (
                    timestamp.split("_")[0] + "_" + timestamp.split("_")[1]
                    if len(timestamp.split("_")) >= 2
                    else timestamp
                )
            else:
                main_timestamp = timestamp

            # Delete all associated files
            files_to_delete = [
                os.path.join(SAVE_DIR, filename),  # The processed image itself
                os.path.join(
                    SAVE_DIR, f"vein_{timestamp}_original.jpg"
                ),  # Original image
                os.path.join(
                    SAVE_DIR, f"metadata_{main_timestamp}.json"
                ),  # Metadata file
            ]

            deleted_count = 0
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
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


@app.route("/method_comparison", methods=["POST"])
def method_comparison():
    """Generate comparison images using different vein detection methods"""
    global camera, vein_detector

    if camera is None:
        return jsonify({"success": False, "message": "Camera not initialized"})

    try:
        # Capture a single high-res frame
        frame = camera.capture_high_res()

        if frame is None:
            return jsonify(
                {"success": False, "message": "Failed to capture comparison frame"}
            )

        # Generate processed images with different methods
        results = {}
        methods = ["adaptive", "frangi", "laplacian", "none"]

        for method in methods:
            try:
                if method != "none":
                    processed = vein_detector.detect_veins(frame.copy(), method=method)
                else:
                    processed = frame.copy()  # No processing

                # Save the image temporarily
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"compare_{method}_{timestamp}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(filepath, processed)

                # Store the URL
                results[method] = url_for("static", filename=f"images/{filename}")
            except Exception as e:
                print(f"Error processing with method {method}: {e}")
                results[method] = None

        return jsonify({"success": True, "comparison": results})
    except Exception as e:
        print(f"Error generating method comparison: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/search_images", methods=["POST"])
def search_images():
    """Search images by patient name, date, or other metadata"""
    try:
        data = request.get_json()
        query = data.get("query", "").lower()

        if not query:
            return jsonify({"success": True, "images": []})

        # Get all images
        all_images = []
        for filename in os.listdir(SAVE_DIR):
            if filename.startswith("metadata_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(SAVE_DIR, filename), "r") as f:
                        metadata = json.load(f)

                    timestamp = metadata.get("timestamp", "")

                    # Check if processed and original images exist
                    processed_path = os.path.join(SAVE_DIR, f"vein_{timestamp}.jpg")
                    original_path = os.path.join(
                        SAVE_DIR, f"vein_{timestamp}_original.jpg"
                    )

                    if os.path.exists(processed_path):
                        processed_url = url_for(
                            "static", filename=f"images/vein_{timestamp}.jpg"
                        )
                        original_url = (
                            url_for(
                                "static",
                                filename=f"images/vein_{timestamp}_original.jpg",
                            )
                            if os.path.exists(original_path)
                            else None
                        )

                        all_images.append(
                            {
                                "timestamp": timestamp,
                                "processed_image": processed_url,
                                "original_image": original_url,
                                "metadata": metadata,
                            }
                        )
                except Exception as e:
                    print(f"Error processing metadata file {filename}: {e}")

        # Filter images based on query
        filtered_images = []
        for img in all_images:
            metadata = img.get("metadata", {})

            # Check timestamp
            if query in img.get("timestamp", "").lower():
                filtered_images.append(img)
                continue

            # Check patient info
            patient_info = metadata.get("patient_info", {})
            patient_name = patient_info.get("name", "").lower()
            patient_id = patient_info.get("id", "").lower()

            if query in patient_name or query in patient_id:
                filtered_images.append(img)
                continue

            # Check detection method
            detection_method = metadata.get("detection_method", "").lower()
            if query in detection_method:
                filtered_images.append(img)
                continue

        return jsonify({"success": True, "images": filtered_images})
    except Exception as e:
        print(f"Error searching images: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/calibration_settings", methods=["POST"])
def calibration_settings():
    """Apply calibration settings"""
    global settings

    try:
        data = request.get_json()

        # Update all calibration settings in one go
        settings_update = {
            "led_brightness": int(
                data.get("led_brightness", settings["led_brightness"])
            ),
            "camera_exposure": int(
                data.get("camera_exposure", settings["camera_exposure"])
            ),
            "camera_gain": float(data.get("camera_gain", settings["camera_gain"])),
            "detection_method": data.get(
                "detection_method", settings["detection_method"]
            ),
        }

        # Use the existing update_settings function
        request._cached_json = (
            lambda: settings_update
        )  # Mock the request JSON for update_settings
        response = update_settings()

        return response
    except Exception as e:
        print(f"Error applying calibration settings: {e}")
        return jsonify({"success": False, "message": str(e)})


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


@app.route("/test")
def test_page():
    """Simple test route to check if Flask is working"""
    return "Flask server is running! Try accessing the main page at '/'."


@app.route("/capture_for_comparison", methods=["POST"])
def capture_for_comparison():
    """Capture a frame and process it with multiple detection methods for side-by-side comparison"""
    global camera, vein_detector

    if camera is None:
        return jsonify({"success": False, "message": "Camera not initialized"})

    try:
        # Capture a single high-res frame
        frame = camera.capture_high_res()

        if frame is None:
            return jsonify(
                {"success": False, "message": "Failed to capture comparison frame"}
            )

        # Define the methods to compare
        methods_to_compare = ["adaptive", "frangi", "laplacian", "none"]
        comparison_results = []

        # Process with each method
        for method in methods_to_compare:
            try:
                # Copy the frame to avoid modifying the original
                frame_copy = frame.copy()

                if method != "none":
                    processed = vein_detector.detect_veins(frame_copy, method=method)
                else:
                    # Just ensure it's in the right format for saving
                    if len(frame_copy.shape) == 2:  # Grayscale
                        processed = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
                    else:
                        processed = frame_copy

                # Save the processed image with a timestamp and method
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"compare_{method}_{timestamp}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(filepath, processed)

                # Create a URL for the image
                image_url = url_for("static", filename=f"images/{filename}")

                # Add to results
                method_info = {
                    "name": method,
                    "image_url": image_url,
                    "description": get_method_description(method),
                }
                comparison_results.append(method_info)

            except Exception as e:
                print(f"Error processing with method {method}: {e}")
                # Add a placeholder for failed method
                comparison_results.append(
                    {
                        "name": method,
                        "image_url": url_for(
                            "static", filename="images/error_placeholder.jpg"
                        ),
                        "description": f"Error: {str(e)}",
                    }
                )

        return jsonify(
            {"success": True, "methods": comparison_results, "timestamp": timestamp}
        )
    except Exception as e:
        print(f"Error generating method comparison: {e}")
        return jsonify({"success": False, "message": str(e)})


def get_method_description(method):
    """Return a description for each detection method"""
    descriptions = {
        "adaptive": "Best for general use and medium to light skin tones. Good balance of detail and contrast.",
        "frangi": "Optimized for thin veins and fine vascular structures. Works well for deeper veins.",
        "laplacian": "Excellent for edge detection and high contrast. Best for surface veins.",
        "none": "Raw unprocessed image showing the original infrared capture.",
    }
    return descriptions.get(method, "No description available")


@app.route("/get_patient_history", methods=["POST"])
def get_patient_history():
    """Retrieve patient history and previous images"""
    try:
        data = request.get_json()
        patient_id = data.get("patient_id", "")

        if not patient_id:
            return jsonify({"success": False, "message": "Patient ID is required"})

        # Search for patient in patient data directory
        patient_files = []
        for filename in os.listdir(PATIENT_DATA_DIR):
            if filename.startswith(f"{patient_id}_") and filename.endswith(".json"):
                patient_files.append(filename)

        if not patient_files:
            return jsonify(
                {"success": False, "message": "No records found for this patient"}
            )

        # Load the most recent patient data
        patient_files.sort(reverse=True)  # Sort by filename (contains timestamp)
        with open(os.path.join(PATIENT_DATA_DIR, patient_files[0]), "r") as f:
            patient_data = json.load(f)

        # Find images associated with this patient
        patient_images = []
        for filename in os.listdir(SAVE_DIR):
            if filename.startswith("metadata_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(SAVE_DIR, filename), "r") as f:
                        metadata = json.load(f)

                    # Check if this image belongs to the patient
                    if metadata.get("patient_info", {}).get("id", "") == patient_id:
                        timestamp = metadata.get("timestamp", "")

                        # Check if image files exist
                        processed_path = f"vein_{timestamp}.jpg"
                        if os.path.exists(os.path.join(SAVE_DIR, processed_path)):
                            patient_images.append(
                                {
                                    "timestamp": timestamp,
                                    "processed_image": url_for(
                                        "static", filename=f"images/{processed_path}"
                                    ),
                                    "detection_method": metadata.get(
                                        "detection_method", "unknown"
                                    ),
                                    "procedure": metadata.get("patient_info", {}).get(
                                        "procedure", ""
                                    ),
                                }
                            )
                except Exception as e:
                    print(f"Error processing metadata file {filename}: {e}")
                    continue

        # Sort images by timestamp (newest first)
        patient_images.sort(
            key=lambda x: x["timestamp"] if x["timestamp"] else "", reverse=True
        )

        return jsonify(
            {
                "success": True,
                "patient_data": patient_data,
                "image_count": len(patient_images),
                "images": patient_images,
            }
        )
    except Exception as e:
        print(f"Error retrieving patient history: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/check_connectivity")
def check_connectivity():
    """Check connectivity of all hardware components"""
    global camera, led_controller

    try:
        camera_ok = camera is not None
        led_ok = led_controller is not None and led_controller.connected

        # More detailed status checks
        camera_details = {
            "connected": camera_ok,
            "simulation": (
                hasattr(camera, "simulation") and camera.simulation if camera else True
            ),
            "resolution": f"{camera.width}x{camera.height}" if camera else "Unknown",
            "frame_rate": camera.frame_rate if camera else 0,
        }

        led_details = {
            "connected": led_ok,
            "port": (
                getattr(led_controller, "port", "Unknown") if led_controller else "None"
            ),
            "brightness": settings["led_brightness"] if led_ok else 0,
            "pattern": settings["led_pattern"] if led_ok else 0,
        }

        # System metrics - simplified for cross-platform compatibility
        system_metrics = {
            "uptime": (
                "Not available on this platform"
                if sys.platform != "linux"
                else "Available"
            ),
            "temperature": (
                "Not available on this platform"
                if sys.platform != "linux"
                else "Available"
            ),
            "disk_free": get_disk_space_percentage(),
        }

        return jsonify(
            {
                "success": True,
                "all_systems_go": camera_ok and led_ok,
                "camera": camera_details,
                "led_controller": led_details,
                "system": system_metrics,
            }
        )
    except Exception as e:
        print(f"Error checking connectivity: {e}")
        return jsonify({"success": False, "message": str(e)})


def get_disk_space_percentage():
    """Get percentage of free disk space"""
    try:
        if sys.platform == "win32":
            import ctypes

            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p("."),
                None,
                ctypes.pointer(total_bytes),
                ctypes.pointer(free_bytes),
            )
            return round((free_bytes.value / total_bytes.value) * 100, 1)
        else:
            # Unix-like systems including macOS and Linux
            st = os.statvfs(".")
            free = st.f_bavail * st.f_frsize
            total = st.f_blocks * st.f_frsize
            return round((free / total) * 100, 1)
    except Exception as e:
        print(f"Error getting disk space: {e}")
        return 0


@app.route("/optimize_auto", methods=["POST"])
def optimize_auto():
    """Automatically optimize camera settings based on current frame analysis"""
    global camera, settings

    if camera is None:
        return jsonify({"success": False, "message": "Camera not initialized"})

    try:
        # Get current frame
        frame = camera.read()

        if frame is None:
            return jsonify({"success": False, "message": "Failed to capture frame"})

        # Analyze frame to determine optimal settings
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame
        )

        # Calculate histogram to determine brightness distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Calculate mean brightness
        mean_val = cv2.mean(gray)[0]

        # Calculate optimal exposure based on brightness
        # Target a middle brightness around 127-140 for optimal vein detection
        current_exposure = settings["camera_exposure"]
        target_brightness = 135

        if mean_val < 70:  # Too dark
            new_exposure = min(
                current_exposure * 1.5, 100000
            )  # Increase exposure, max 100ms
            new_gain = min(settings["camera_gain"] * 1.3, 16)  # Increase gain, max 16
        elif mean_val > 200:  # Too bright
            new_exposure = max(
                current_exposure * 0.7, 1000
            )  # Decrease exposure, min 1ms
            new_gain = max(settings["camera_gain"] * 0.7, 1)  # Decrease gain, min 1
        else:
            # Fine-tune: adjust to get closer to target
            adjustment_factor = target_brightness / mean_val
            new_exposure = max(min(current_exposure * adjustment_factor, 100000), 1000)
            new_gain = settings[
                "camera_gain"
            ]  # Keep gain the same for fine adjustments

        # Update settings
        updated_settings = {
            "camera_exposure": int(new_exposure),
            "camera_gain": float(new_gain),
        }

        # Also adjust CLAHE parameters based on image analysis
        # Calculate image contrast
        stddev = np.std(gray)

        # If low contrast, increase CLAHE clip limit
        if stddev < 30:
            updated_settings["clahe_clip_limit"] = min(
                settings["clahe_clip_limit"] * 1.3, 10
            )
        elif stddev > 60:
            updated_settings["clahe_clip_limit"] = max(
                settings["clahe_clip_limit"] * 0.7, 1
            )

        # Apply the new settings
        for key, value in updated_settings.items():
            settings[key] = value

        # Update camera with new settings
        if camera and not camera.simulation:
            camera.adjust_settings(
                exposure=settings["camera_exposure"], gain=settings["camera_gain"]
            )

        # Update vein detector parameters
        detector_params = {
            "clahe_clip_limit": settings["clahe_clip_limit"],
            "clahe_tile_grid_size": settings["clahe_tile_grid_size"],
        }
        vein_detector.update_params(detector_params)

        return jsonify(
            {
                "success": True,
                "message": "Settings automatically optimized",
                "updated_settings": updated_settings,
                "analysis": {
                    "mean_brightness": mean_val,
                    "contrast_stddev": stddev,
                    "target_brightness": target_brightness,
                },
            }
        )
    except Exception as e:
        print(f"Error optimizing settings: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/export_data", methods=["POST"])
def export_data():
    """Export images and data for a specific patient or time range"""
    try:
        data = request.get_json()
        export_type = data.get("type", "patient")  # 'patient' or 'timerange'

        if export_type == "patient":
            patient_id = data.get("patient_id", "")
            if not patient_id:
                return jsonify({"success": False, "message": "Patient ID is required"})

            # Create export directory
            export_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "static",
                    "exports",
                    f"patient_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
            )
            os.makedirs(export_dir, exist_ok=True)

            # Collect patient data
            patient_data = {}
            patient_files = []
            for filename in os.listdir(PATIENT_DATA_DIR):
                if filename.startswith(f"{patient_id}_") and filename.endswith(".json"):
                    patient_files.append(filename)

            if patient_files:
                patient_files.sort(reverse=True)
                with open(os.path.join(PATIENT_DATA_DIR, patient_files[0]), "r") as f:
                    patient_data = json.load(f)

                # Write patient data to export
                with open(os.path.join(export_dir, "patient_info.json"), "w") as f:
                    json.dump(patient_data, f, indent=2)

            # Find and copy images for this patient
            image_count = 0
            for filename in os.listdir(SAVE_DIR):
                if filename.startswith("metadata_") and filename.endswith(".json"):
                    try:
                        with open(os.path.join(SAVE_DIR, filename), "r") as f:
                            metadata = json.load(f)

                        if metadata.get("patient_info", {}).get("id", "") == patient_id:
                            # Copy metadata file
                            shutil.copy2(
                                os.path.join(SAVE_DIR, filename),
                                os.path.join(export_dir, filename),
                            )

                            # Copy image files
                            timestamp = metadata.get("timestamp", "")
                            processed_path = f"vein_{timestamp}.jpg"
                            original_path = f"vein_{timestamp}_original.jpg"

                            if os.path.exists(os.path.join(SAVE_DIR, processed_path)):
                                shutil.copy2(
                                    os.path.join(SAVE_DIR, processed_path),
                                    os.path.join(export_dir, processed_path),
                                )
                                image_count += 1

                            if os.path.exists(os.path.join(SAVE_DIR, original_path)):
                                shutil.copy2(
                                    os.path.join(SAVE_DIR, original_path),
                                    os.path.join(export_dir, original_path),
                                )
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
                        continue

            # Create HTML report
            create_html_report(export_dir, patient_data, image_count)

            return jsonify(
                {
                    "success": True,
                    "message": f"Exported {image_count} images and data for patient {patient_id}",
                    "export_directory": export_dir,
                    "patient_name": patient_data.get("name", "Unknown"),
                    "image_count": image_count,
                }
            )

        elif export_type == "timerange":
            start_date = data.get("start_date")
            end_date = data.get("end_date")

            if not start_date or not end_date:
                return jsonify(
                    {"success": False, "message": "Start and end dates are required"}
                )

            # Parse dates
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                end = end.replace(hour=23, minute=59, second=59)  # End of day
            except ValueError:
                return jsonify(
                    {"success": False, "message": "Invalid date format. Use YYYY-MM-DD"}
                )

            # Create export directory
            export_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "static",
                    "exports",
                    f"timerange_{start_date}_to_{end_date}",
                )
            )
            os.makedirs(export_dir, exist_ok=True)

            # Find images in the time range
            image_count = 0
            for filename in os.listdir(SAVE_DIR):
                if filename.startswith("vein_") and not filename.endswith(
                    "_original.jpg"
                ):
                    try:
                        # Extract timestamp from filename (YYYYMMDD_HHMMSS)
                        timestamp_str = filename.replace("vein_", "").split(".")[0]
                        file_date = datetime.strptime(
                            timestamp_str.split("_")[0], "%Y%m%d"
                        )

                        # Check if in range
                        if start <= file_date <= end:
                            # Copy the processed image
                            shutil.copy2(
                                os.path.join(SAVE_DIR, filename),
                                os.path.join(export_dir, filename),
                            )
                            image_count += 1

                            # Copy the original image if it exists
                            original = filename.replace(".jpg", "_original.jpg")
                            if os.path.exists(os.path.join(SAVE_DIR, original)):
                                shutil.copy2(
                                    os.path.join(SAVE_DIR, original),
                                    os.path.join(export_dir, original),
                                )

                            # Copy metadata if it exists
                            metadata = f"metadata_{timestamp_str}.json"
                            if os.path.exists(os.path.join(SAVE_DIR, metadata)):
                                shutil.copy2(
                                    os.path.join(SAVE_DIR, metadata),
                                    os.path.join(export_dir, metadata),
                                )
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
                        continue

            return jsonify(
                {
                    "success": True,
                    "message": f"Exported {image_count} images from {start_date} to {end_date}",
                    "export_directory": export_dir,
                    "start_date": start_date,
                    "end_date": end_date,
                    "image_count": image_count,
                }
            )

        else:
            return jsonify({"success": False, "message": "Invalid export type"})
    except Exception as e:
        print(f"Error exporting data: {e}")
        return jsonify({"success": False, "message": str(e)})


def create_html_report(export_dir, patient_data, image_count):
    """Create an HTML report of the exported data"""
    try:
        report_path = os.path.join(export_dir, "report.html")

        # Get list of images in export dir
        images = []
        for filename in os.listdir(export_dir):
            if filename.startswith("vein_") and not filename.endswith("_original.jpg"):
                images.append(filename)

        # Sort images by timestamp (newest first)
        images.sort(reverse=True)

        # Build HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>VeinVision Pro - Patient Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #052c65, #0d6efd); color: white; padding: 20px; border-radius: 10px; }}
                .patient-info {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .gallery {{ display: flex; flex-wrap: wrap; gap: 15px; }}
                .image-card {{ border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 300px; }}
                .image-card img {{ width: 100%; height: auto; }}
                .image-info {{ padding: 10px; background: #f5f5f5; }}
                h1, h2, h3 {{ margin-top: 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>VeinVision Pro - Patient Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            </div>
            
            <div class="patient-info">
                <h2>Patient Information</h2>
                <table>
                    <tr>
                        <th>Name:</th>
                        <td>{patient_data.get('name', 'Not recorded')}</td>
                    </tr>
                    <tr>
                        <th>ID:</th>
                        <td>{patient_data.get('id', 'Not recorded')}</td>
                    </tr>
                    <tr>
                        <th>Age:</th>
                        <td>{patient_data.get('age', 'Not recorded')}</td>
                    </tr>
                    <tr>
                        <th>Procedure:</th>
                        <td>{patient_data.get('procedure', 'Not recorded')}</td>
                    </tr>
                </table>
            </div>
            
            <h2>Captured Images ({image_count})</h2>
            <div class="gallery">
        """

        # Add images to the report
        for image_file in images[:12]:  # Limit to 12 images for performance
            try:
                # Extract timestamp from filename
                timestamp_str = image_file.replace("vein_", "").split(".")[0]

                # Try to load metadata
                metadata_file = f"metadata_{timestamp_str}.json"
                metadata_path = os.path.join(export_dir, metadata_file)

                detection_method = "Unknown"
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        detection_method = metadata.get(
                            "detection_method", "Unknown"
                        ).capitalize()

                # Format date for display
                try:
                    date_part = timestamp_str.split("_")[0]
                    time_part = timestamp_str.split("_")[1]
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
                except:
                    formatted_date = timestamp_str

                html += f"""
                <div class="image-card">
                    <img src="{image_file}" alt="Vein image from {formatted_date}">
                    <div class="image-info">
                        <p><strong>Date:</strong> {formatted_date}</p>
                        <p><strong>Method:</strong> {detection_method}</p>
                    </div>
                </div>
                """
            except Exception as e:
                print(f"Error adding image {image_file} to report: {e}")
                continue

        # Complete the HTML
        html += """
            </div>
            
            <div style="margin-top: 30px; text-align: center; color: #666; font-size: 0.8em;">
                <p>VeinVision Pro - Advanced Medical Imaging System for Venipuncture Assistance</p>
            </div>
        </body>
        </html>
        """

        # Write the HTML report
        with open(report_path, "w") as f:
            f.write(html)

        print(f"HTML report created at {report_path}")
        return True
    except Exception as e:
        print(f"Error creating HTML report: {e}")
        return False


@app.route("/get_configurations")
def get_configurations():
    """Get all available presets and configurations"""
    try:
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))
        presets = []

        # Create configs directory if it doesn't exist
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

            # Create default presets
            default_presets = [
                {
                    "name": "Standard",
                    "description": "Balanced settings for most patients",
                    "settings": {
                        "detection_method": "adaptive",
                        "camera_exposure": 20000,
                        "camera_gain": 6.0,
                        "clahe_clip_limit": 5.0,
                        "clahe_tile_grid_size": 8,
                    },
                },
                {
                    "name": "High Contrast",
                    "description": "Enhanced contrast for difficult veins",
                    "settings": {
                        "detection_method": "frangi",
                        "camera_exposure": 30000,
                        "camera_gain": 8.0,
                        "clahe_clip_limit": 7.0,
                        "clahe_tile_grid_size": 8,
                    },
                },
                {
                    "name": "Deep Veins",
                    "description": "Optimized for deeper veins",
                    "settings": {
                        "detection_method": "frangi",
                        "camera_exposure": 40000,
                        "camera_gain": 10.0,
                        "frangi_scale_min": 1.0,
                        "frangi_scale_max": 10.0,
                        "frangi_beta": 0.6,
                        "frangi_gamma": 20,
                    },
                },
                {
                    "name": "Pediatric",
                    "description": "Gentle settings for children",
                    "settings": {
                        "detection_method": "adaptive",
                        "camera_exposure": 15000,
                        "camera_gain": 5.0,
                        "led_brightness": 180,
                    },
                },
            ]

            # Save default presets
            for preset in default_presets:
                with open(
                    os.path.join(
                        config_dir, f"{preset['name'].lower().replace(' ', '_')}.json"
                    ),
                    "w",
                ) as f:
                    json.dump(preset, f, indent=2)
                presets.append(preset)
        else:
            # Load existing presets
            for filename in os.listdir(config_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(config_dir, filename), "r") as f:
                            preset = json.load(f)
                            presets.append(preset)
                    except Exception as e:
                        print(f"Error loading preset {filename}: {e}")
                        continue

        return jsonify({"success": True, "presets": presets})
    except Exception as e:
        print(f"Error getting configurations: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/save_configuration", methods=["POST"])
def save_configuration():
    """Save a new configuration preset"""
    try:
        data = request.get_json()
        name = data.get("name", "")
        description = data.get("description", "")

        if not name:
            return jsonify({"success": False, "message": "Preset name is required"})

        # Get settings to save
        preset_settings = {}
        for key in [
            "detection_method",
            "camera_exposure",
            "camera_gain",
            "clahe_clip_limit",
            "clahe_tile_grid_size",
            "led_brightness",
            "led_pattern",
            "frangi_scale_min",
            "frangi_scale_max",
            "frangi_beta",
            "frangi_gamma",
        ]:
            if key in settings:
                preset_settings[key] = settings[key]

        # Create preset object
        preset = {
            "name": name,
            "description": description,
            "settings": preset_settings,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save to file
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        filename = f"{name.lower().replace(' ', '_')}.json"
        with open(os.path.join(config_dir, filename), "w") as f:
            json.dump(preset, f, indent=2)

        return jsonify(
            {
                "success": True,
                "message": f"Configuration '{name}' saved successfully",
                "preset": preset,
            }
        )
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/load_configuration", methods=["POST"])
def load_configuration():
    """Load a saved configuration preset"""
    global settings, camera, vein_detector

    try:
        data = request.get_json()
        name = data.get("name", "")

        if not name:
            return jsonify({"success": False, "message": "Preset name is required"})

        # Find and load the preset file
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))
        filename = f"{name.lower().replace(' ', '_')}.json"
        filepath = os.path.join(config_dir, filename)

        if not os.path.exists(filepath):
            return jsonify({"success": False, "message": f"Preset '{name}' not found"})

        # Load the preset
        with open(filepath, "r") as f:
            preset = json.load(f)

        # Extract settings
        preset_settings = preset.get("settings", {})

        # Update local settings
        for key, value in preset_settings.items():
            if key in settings:
                settings[key] = value

        # Apply settings to hardware
        # Camera settings
        if camera and not camera.simulation:
            camera.adjust_settings(
                exposure=settings["camera_exposure"], gain=settings["camera_gain"]
            )

        # LED settings
        if led_controller and led_controller.connected:
            led_controller.set_brightness(settings["led_brightness"])
            led_controller.set_pattern(settings["led_pattern"])

        # Update vein detector parameters
        detector_params = {
            k: v
            for k, v in settings.items()
            if k
            in [
                "clahe_clip_limit",
                "clahe_tile_grid_size",
                "frangi_scale_min",
                "frangi_scale_max",
                "frangi_scale_step",
                "frangi_beta",
                "frangi_gamma",
            ]
        }
        vein_detector.update_params(detector_params)

        return jsonify(
            {
                "success": True,
                "message": f"Configuration '{name}' loaded successfully",
                "settings": settings,
            }
        )
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return jsonify({"success": False, "message": str(e)})


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
