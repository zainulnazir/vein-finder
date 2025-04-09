#!/usr/bin/env python3

"""
Vein Finder - System Check Utility

This script checks if all required components and libraries are available
and functioning correctly for the Vein Finder application.
"""

import os
import sys
import time
import importlib
import subprocess
from pathlib import Path

def check_module(module_name, package_name=None):
    """Check if a Python module is installed"""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed")
        print(f"   Install with: pip install {package_name}")
        return False

def check_camera():
    """Check if the Raspberry Pi camera is connected and accessible"""
    try:
        # Try to import picamera2
        from picamera2 import Picamera2
        
        # Try to initialize camera with the working configuration
        camera = Picamera2()
        camera_config = camera.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        camera.configure(camera_config)
        camera.start()
        
        # Capture a test frame
        time.sleep(2)  # Allow camera to adjust
        test_frame = camera.capture_array()
        camera.stop()
        
        if test_frame is not None and test_frame.size > 0:
            print(f"✅ Camera is connected and working")
            
            # Create test directory if it doesn't exist
            os.makedirs("test_output", exist_ok=True)
            
            # Try to save test image using OpenCV
            try:
                import cv2
                test_path = os.path.join("test_output", "camera_test.jpg")
                
                # Convert to BGR for OpenCV if needed
                if len(test_frame.shape) == 3 and test_frame.shape[2] == 3:
                    test_frame_cv = cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR)
                else:
                    test_frame_cv = test_frame
                    
                cv2.imwrite(test_path, test_frame_cv)
                print(f"✅ Test image saved to {test_path}")
            except Exception as e:
                print(f"⚠️ Camera works but couldn't save test image: {e}")
            
            return True
        else:
            print("❌ Camera connected but couldn't capture frame")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

def check_arduino():
    """Check if Arduino is connected"""
    import serial.tools.list_ports
    
    ports = list(serial.tools.list_ports.comports())
    arduino_port = None
    
    # Look for Arduino
    for port in ports:
        if "Arduino" in port.description or "ACM" in port.device or "USB" in port.device:
            arduino_port = port.device
            print(f"✅ Arduino found on port {arduino_port}")
            
            # Try to communicate with Arduino
            try:
                ser = serial.Serial(arduino_port, 9600, timeout=2)
                time.sleep(2)  # Wait for Arduino to reset
                
                # Try to send a command and get response
                ser.write(b"T1\n")  # Turn on LEDs
                response = ser.readline().decode().strip()
                
                if response == "OK":
                    print("✅ Arduino communication successful")
                else:
                    print(f"⚠️ Arduino responded with: {response}")
                    
                # Turn LEDs off
                ser.write(b"T0\n")
                ser.close()
                
            except Exception as e:
                print(f"⚠️ Arduino found but communication failed: {e}")
                
            return True
    
    print("❌ Arduino not found")
    print("   Make sure Arduino is connected via USB and has the correct sketch uploaded")
    print("   Available ports: " + ", ".join([p.device for p in ports]) if ports else "None")
    return False

def check_directories():
    """Check if all required directories exist"""
    required_dirs = [
        "static",
        "static/images",
        "templates",
        "arduino_sketch"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            all_exist = False
            # Create the directory
            os.makedirs(directory, exist_ok=True)
            print(f"   Created directory: {directory}")
    
    return all_exist

def check_files():
    """Check if all required files exist"""
    required_files = [
        "app.py",
        "camera.py",
        "vein_detection.py",
        "led_controller.py",
        "requirements.txt",
        "arduino_sketch/vein_finder_led_control.ino",
        "templates/index.html"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            all_exist = False
    
    return all_exist

def check_permissions():
    """Check if the script has necessary permissions"""
    # Check if we can write to the directories
    try:
        test_file = os.path.join("static", "test_permissions.txt")
        with open(test_file, 'w') as f:
            f.write("Test")
        os.remove(test_file)
        print("✅ Write permissions OK")
        return True
    except Exception as e:
        print(f"❌ Permission error: {e}")
        return False

def check_dependencies():
    """Check for required Python packages"""
    required_modules = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "flask": "flask",
        "picamera2": "picamera2",
        "serial": "pyserial",
        "skimage": "scikit-image"
    }
    
    all_installed = True
    for module, package in required_modules.items():
        if not check_module(module, package):
            all_installed = False
    
    return all_installed

def display_system_info():
    """Display system information"""
    print("\n=== System Information ===")
    
    # OS information
    try:
        os_info = subprocess.check_output(["uname", "-a"]).decode().strip()
        print(f"OS: {os_info}")
    except:
        print("OS: Unknown")
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Raspberry Pi model
    try:
        with open("/proc/device-tree/model", "r") as f:
            pi_model = f.read().strip()
            print(f"Raspberry Pi Model: {pi_model}")
    except:
        print("Raspberry Pi Model: Unknown or not running on a Raspberry Pi")
    
    # Check for GPU memory
    try:
        gpu_mem = subprocess.check_output(["vcgencmd", "get_mem", "gpu"]).decode().strip()
        print(f"GPU Memory: {gpu_mem}")
    except:
        print("GPU Memory: Unknown")

def main():
    """Main function to run all checks"""
    print("=== Vein Finder - System Check ===")
    print("Checking if the system is properly set up...\n")
    
    # Display system information
    display_system_info()
    
    # Check dependencies
    print("\n=== Checking Python Dependencies ===")
    dependencies_ok = check_dependencies()
    
    # Check directories and files
    print("\n=== Checking Directories and Files ===")
    directories_ok = check_directories()
    files_ok = check_files()
    
    # Check permissions
    print("\n=== Checking Permissions ===")
    permissions_ok = check_permissions()
    
    # Check hardware if dependencies are installed
    hardware_checks = True
    if dependencies_ok:
        print("\n=== Checking Hardware ===")
        
        print("\n--- Camera Check ---")
        camera_ok = check_camera()
        
        print("\n--- Arduino Check ---")
        arduino_ok = check_arduino()
        
        hardware_checks = camera_ok and arduino_ok
    
    # Summary
    print("\n=== Summary ===")
    
    checks = [
        ("Dependencies", dependencies_ok),
        ("Directories", directories_ok),
        ("Files", files_ok),
        ("Permissions", permissions_ok),
        ("Hardware", hardware_checks)
    ]
    
    all_ok = True
    for check_name, status in checks:
        status_str = "✅ OK" if status else "❌ Issues found"
        print(f"{check_name}: {status_str}")
        if not status:
            all_ok = False
    
    if all_ok:
        print("\n✅ All checks passed! The system is ready to run the Vein Finder application.")
        print("   Start the application with: python app.py")
    else:
        print("\n⚠️ Some checks failed. Please fix the issues before running the application.")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 