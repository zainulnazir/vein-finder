#!/usr/bin/env python3

import time
import os
import cv2
import numpy as np
from threading import Thread
from datetime import datetime
import sys
import platform


class VeinCamera:
    def __init__(self, resolution=(640, 480), framerate=30, simulation_mode=None):
        """Initialize the camera with IR optimization for vein detection

        Args:
            resolution: Tuple of (width, height) - default is 480p (4:3)
            framerate: Target framerate
            simulation_mode: Force simulation mode ('vein_sim', 'gradient', or None for auto-detection)
        """
        self.resolution = resolution
        self.framerate = framerate
        self.frame = None
        self.stopped = False
        # Use absolute path for image saves to avoid CWD/permission issues
        self.images_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "static", "images")
        )
        self.simulation = simulation_mode
        # Available resolutions with appropriate framerates (prefer 4:3).
        # Keep UI-friendly names but map 16:9 options to closest 4:3 to avoid FoV/cropping issues on IMX219.
        self.available_resolutions = {
            "480p": {"resolution": (640, 480), "framerate": 30},
            "720p": {"resolution": (1024, 768), "framerate": 30},   # Map 720p UI to 1024x768 (4:3)
            "1080p": {"resolution": (1640, 1232), "framerate": 20}, # Map 1080p UI to 1640x1232 (4:3)
            "768p": {"resolution": (1024, 768), "framerate": 30},
            "1232p": {"resolution": (1640, 1232), "framerate": 20},
        }
        # Highest resolution for captures (Using 4:3)
        self.capture_resolution = self.available_resolutions["1232p"]["resolution"] # Use 1640x1232 for high-res capture

        # Create directory for saved images if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)

        # Check if we should use simulation mode (if not explicitly set)
        if self.simulation is None:
            # Auto-detect if not running on a Raspberry Pi
            self.simulation = not self._is_raspberry_pi()

        if self.simulation:
            print("Running in camera simulation mode")
        else:
            try:
                # Import here to avoid errors on non-RPi systems
                from picamera2 import Picamera2

                self.picam2 = Picamera2()
                self._configure_camera()
            except ImportError:
                print("PiCamera2 not available, switching to simulation mode")
                self.simulation = True
            except Exception as e:
                print(f"Camera initialization error: {e}")
                print("Switching to simulation mode")
                self.simulation = True

    def _configure_camera(self):
        """Configure camera with current resolution settings"""
        if self.simulation:
            return

        try:
            # Configure camera for video mode with current resolution
            # Define the full sensor crop region
            # full_sensor_crop = (0, 0, 3280, 2464) # Use full sensor area -- Removing ScalerCrop attempts

            # 1. Create configuration specifying MAIN and RAW streams
            # Requesting raw stream often forces a specific sensor mode (e.g., full FoV binned)
            raw_size = (1640, 1232) # IMX219 2x2 binned mode (full FoV)
            print(f"Requesting main={self.resolution} and raw={raw_size} to encourage full FoV.")
            config = self.picam2.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                # lores={"size": (320, 240), "format": "YUV420"}, # Removing lores for simplicity
                raw={"size": raw_size}
            )

            # 2. Configure the camera
            self.picam2.configure(config)
            print(f"Configured camera for {self.resolution} with raw stream hint.")

            # 3. Set ScalerCrop AFTER configuration ## REMOVED ## START
            # try:
            #      print(f"Attempting to set ScalerCrop post-config: {full_sensor_crop}")
            #      self.picam2.set_controls({"ScalerCrop": full_sensor_crop})
            #      print("ScalerCrop set successfully post-config.")
            # except Exception as e:
            #      print(f"Warning: Failed to set ScalerCrop post-config: {e}")
            # 3. Set ScalerCrop AFTER configuration ## REMOVED ## END

            # Set OTHER camera parameters optimal for IR imaging (AFTER configure)
            try:
                self.picam2.set_controls(
                    {
                        "ExposureTime": 20000,  # Higher exposure for IR
                        "AnalogueGain": 6.0,  # Gain setting from working example
                        "Saturation": 1.0,  # Default saturation
                        "Sharpness": 1.0,  # Default sharpness
                    }
                )
            except Exception as e:
                 print(f"Warning: Could not set post-config controls (Exposure/Gain etc): {e}")

            # Try to set noise reduction if available
            try:
                self.picam2.set_controls(
                    {
                        "NoiseReductionMode": 2
                    }  # Minimal noise reduction to preserve details
                )
            except:
                print("Warning: Noise reduction setting not available")
        except Exception as e:
            print(f"Warning: Using simple camera configuration due to: {e}")
            # Fallback to basic configuration
            try:
                config = self.picam2.create_video_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.picam2.configure(config)
            except:
                print("Error configuring camera with basic settings")

    def set_resolution(self, resolution_name):
        """Change the camera resolution on the fly

        Args:
            resolution_name: The resolution to use ("480p", "720p", "1080p")

        Returns:
            bool: True if successful, False otherwise
        """
        if resolution_name not in self.available_resolutions:
            print(f"Invalid resolution: {resolution_name}")
            return False

        res_config = self.available_resolutions[resolution_name]
        self.resolution = res_config["resolution"]
        self.framerate = res_config["framerate"]

        if not self.simulation:
            try:
                # Need to stop and restart camera to change resolution
                was_running = hasattr(self, "picam2") and getattr(
                    self.picam2, "_running", False
                )
                if was_running:
                    self.picam2.stop()

                self._configure_camera()

                if was_running:
                    self.picam2.start()
                    # Short wait for camera to initialize with new settings
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error changing resolution: {e}")
                return False

        print(
            f"Resolution changed to {resolution_name}: {self.resolution}, {self.framerate}fps"
        )
        return True

    def capture_high_res(self):
        """Capture a single frame at the highest resolution

        Returns:
            numpy.ndarray: The captured high-resolution image
        """
        if self.simulation:
            # Generate a simulated frame at high resolution
            original_resolution = self.resolution
            self.resolution = self.capture_resolution
            high_res_frame = self._generate_simulated_frame()
            self.resolution = original_resolution
            return high_res_frame

        # For real camera
        try:
            # Store current resolution
            current_resolution = self.resolution

            # Calculate target aspect ratio
            # target_aspect = current_resolution[0] / current_resolution[1] # Not needed

            # Define the full sensor crop region
            # full_sensor_crop = (0, 0, 3280, 2464) # Removed

            # Temporarily switch to high resolution for capture
            # Use the desired capture resolution and matching raw stream size
            raw_size = (1640, 1232) # Match capture res for consistency
            print(f"Requesting still main={self.capture_resolution} and raw={raw_size}.")
            config = self.picam2.create_still_configuration(
                main={"size": self.capture_resolution, "format": "RGB888"}, # Use 4:3 high-res (1640x1232)
                raw={"size": raw_size}
            )
            # Capture the returned array from switch_mode_and_capture_array
            high_res_frame = self.picam2.switch_mode_and_capture_array(config)

            if high_res_frame is None:
                raise ValueError("switch_mode_and_capture_array returned None")
            print(f"High-res frame captured with shape: {high_res_frame.shape}") # Add log

            # Return to the previous configuration (video)
            # Recreate video config using the same logic as _configure_camera
            raw_size_video = (1640, 1232)
            video_config = self.picam2.create_video_configuration(
                main={"size": current_resolution, "format": "RGB888"},
                raw={"size": raw_size_video}
            )
            self.picam2.switch_mode(video_config)

            return high_res_frame
        except Exception as e:
            print(f"Error capturing high-res image: {e}")
            # Fallback to current resolution
            return self.read()

    def _is_raspberry_pi(self):
        """Check if running on a Raspberry Pi"""
        # Check for Raspberry Pi specific file
        if platform.system() == "Linux" and os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                model = f.read()
                if "Raspberry Pi" in model:
                    return True
        return False

    def start(self):
        """Start the camera and capture thread"""
        if not self.simulation:
            # Start real camera
            try:
                self.picam2.start()
                # Wait for camera to initialize
                time.sleep(2)
            except Exception as e:
                print(f"Error starting camera: {e}")
                print("Switching to simulation mode")
                self.simulation = True

        # Start the thread to read frames from the camera or generate simulated frames
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """Update frame in background thread"""
        while not self.stopped:
            if self.simulation:
                # Generate a simulated frame
                self.frame = self._generate_simulated_frame()
                # Add realistic frame rate
                time.sleep(1.0 / self.framerate)
            else:
                # Capture frame from real camera
                try:
                    self.frame = self.picam2.capture_array()
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    # Generate a blank frame or simulated frame on error
                    self.frame = self._generate_simulated_frame()
                    time.sleep(0.1)

        # Stop the camera when thread is stopped
        if not self.simulation:
            try:
                self.picam2.stop()
            except:
                pass

    def _generate_simulated_frame(self):
        """Generate a simulated frame for testing without real camera hardware"""
        width, height = self.resolution

        # Create a grayscale gradient image as base
        gradient = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            gradient[i, :] = int(255 * (i / height))

        # Add simulated vein patterns
        frame = gradient.copy()

        # Add some noise for realism
        noise = np.random.normal(0, 5, (height, width)).astype(np.int16)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        # Add simulated veins (dark lines)
        center_y = height // 2
        thickness = height // 20
        curve_amount = width // 4

        # Create some curved vein-like patterns
        for i in range(3):  # Add 3 simulated veins
            offset = (i - 1) * height // 4
            for x in range(width):
                # Calculate a curved line
                sin_factor = np.sin(x * (6.28 / width)) * curve_amount
                y_pos = center_y + offset + int(sin_factor)
                if 0 <= y_pos < height:
                    # Draw a dark line with gaussian profile
                    for y in range(
                        max(0, y_pos - thickness), min(height, y_pos + thickness)
                    ):
                        # Calculate distance from center of vein
                        dist = abs(y - y_pos)
                        if dist < thickness:
                            # Darken based on distance from center (gaussian-like)
                            darkening = int(
                                120 * np.exp(-(dist**2) / (2 * (thickness / 2) ** 2))
                            )
                            frame[y, x] = max(
                                20, frame[y, x] - darkening
                            )  # Limit how dark it can get

        return frame

    def read(self):
        """Return the current frame"""
        return self.frame

    def stop(self):
        """Stop the camera thread"""
        self.stopped = True

    def save_image(self, frame_to_save, suffix=""):
        """Save the given frame to disk with an optional filename suffix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for uniqueness
        base_filename = f"vein_{timestamp}"
        filename = f"{base_filename}{suffix}.jpg"
        filepath = os.path.join(self.images_dir, filename)

        try:
            # Ensure the frame is not None and is a numpy array
            if frame_to_save is not None and isinstance(frame_to_save, np.ndarray):
                # Save with high quality
                cv2.imwrite(filepath, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"Image saved to: {filepath}")
                return filename
            else:
                print("Error: Frame to save is None or not a valid image.")
                return None
        except Exception as e:
            print(f"Error saving image {filepath}: {e}")
            return None

    def adjust_settings(self, exposure=None, gain=None):
        """Adjust camera settings for optimal vein imaging"""
        if self.simulation:
            # Just store the values in simulation mode
            self.exposure = (
                exposure if exposure is not None else getattr(self, "exposure", 20000)
            )
            self.gain = gain if gain is not None else getattr(self, "gain", 6.0)
            return

        controls = {}

        if exposure is not None:
            controls["ExposureTime"] = int(exposure)

        if gain is not None:
            controls["AnalogueGain"] = float(gain)

        if controls:
            try:
                self.picam2.set_controls(controls)
            except Exception as e:
                print(f"Error adjusting camera settings: {e}")

    def get_camera_name(self):
        """Get camera model name"""
        if self.simulation:
            return "Simulated Camera"

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
