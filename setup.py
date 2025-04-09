#!/usr/bin/env python3

"""
Vein Finder - Setup Script

This script sets up the necessary directories and checks for requirements.
"""

import os
import sys
import subprocess
import platform
import argparse

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        "static",
        "static/images",
        "templates",
        "arduino_sketch",
        "test_output"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"Directory already exists: {directory}")

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version.major < required_version[0] or \
       (current_version.major == required_version[0] and 
        current_version.minor < required_version[1]):
        print(f"Error: Python {required_version[0]}.{required_version[1]}+ is required.")
        print(f"Current Python version: {current_version.major}.{current_version.minor}")
        return False
    
    print(f"Python version OK: {current_version.major}.{current_version.minor}")
    return True

def check_raspberry_pi():
    """Check if running on a Raspberry Pi"""
    system = platform.system()
    if system != "Linux":
        print(f"Warning: This application is designed to run on Raspberry Pi (Linux).")
        print(f"Current system: {system}")
        return False
    
    # Check for Raspberry Pi specific file
    if os.path.exists("/proc/device-tree/model"):
        with open("/proc/device-tree/model", "r") as f:
            model = f.read()
            if "Raspberry Pi" in model:
                print(f"Running on Raspberry Pi: {model.strip()}")
                return True
    
    print("Warning: Not running on a Raspberry Pi.")
    print("The camera and GPIO functionality may not work correctly.")
    return False

def check_system_packages():
    """Check for required system packages"""
    system_packages = [
        "python3-picamera2",
        "python3-opencv",
        "python3-numpy",
        "python3-flask",
        "python3-serial",
        "python3-skimage",
        "python3-matplotlib"
    ]
    
    print("Checking for required system packages...")
    missing_packages = []
    
    try:
        # Check if dpkg is available (Debian/Ubuntu based)
        subprocess.run(["which", "dpkg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for package in system_packages:
            try:
                result = subprocess.run(
                    ["dpkg", "-l", package], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    check=False
                )
                if result.returncode != 0:
                    missing_packages.append(package)
            except Exception:
                missing_packages.append(package)
                
    except subprocess.CalledProcessError:
        print("Warning: Could not check system packages (dpkg not found)")
        return False
        
    if missing_packages:
        print(f"Missing system packages: {', '.join(missing_packages)}")
        print("\nInstall these packages with:")
        print(f"sudo apt install {' '.join(missing_packages)}")
        return False
    else:
        print("All required system packages are installed.")
        return True

def install_requirements(skip_pip=False):
    """Install required Python packages"""
    if skip_pip:
        print("Skipping pip installation as requested.")
        print("Using system-installed packages instead.")
        return True
        
    if os.path.exists("requirements.txt"):
        print("Installing required packages from requirements.txt...")
        try:
            try:
                # First try without breaking system packages
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            except subprocess.CalledProcessError as e:
                # Check if this is due to externally managed environment
                if "externally-managed-environment" in str(e.output) or "externally-managed-environment" in str(e.stderr):
                    print("Detected externally managed environment (PEP 668).")
                    print("Please use system packages instead:")
                    print("sudo apt install python3-picamera2 python3-opencv python3-numpy python3-flask python3-serial python3-skimage python3-matplotlib")
                    return False
                else:
                    raise e
                    
            print("Successfully installed required packages.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to install required packages: {e}")
            print("\nTry using system packages instead:")
            print("sudo apt install python3-picamera2 python3-opencv python3-numpy python3-flask python3-serial python3-skimage python3-matplotlib")
            return False
    else:
        print("Error: requirements.txt not found.")
        return False

def check_camera_module():
    """Check if the camera module is enabled on Raspberry Pi"""
    if not os.path.exists("/proc/device-tree/model"):
        print("Not running on a Raspberry Pi, skipping camera module check.")
        return True
    
    try:
        # Check if camera interface is enabled using vcgencmd
        result = subprocess.run(["vcgencmd", "get_camera"], capture_output=True, text=True)
        if "supported=1 detected=1" in result.stdout:
            print("Camera module is enabled and detected.")
            return True
        else:
            print("Warning: Camera module not detected or not enabled.")
            print("Run 'sudo raspi-config' and ensure the camera interface is enabled.")
            return False
    except FileNotFoundError:
        print("Warning: Could not check camera module status (vcgencmd not found).")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Setup the Vein Finder application')
    parser.add_argument('--skip-pip', action='store_true', help='Skip pip installation and use system packages')
    return parser.parse_args()

def main():
    """Main setup function"""
    args = parse_args()
    
    print("=== Vein Finder - Setup ===")
    print("Note: This application requires system packages for camera support.")
    print("Recommended: 'sudo apt install python3-picamera2 python3-opencv python3-numpy python3-flask python3-serial python3-skimage python3-matplotlib'")
    
    # Create directories
    print("\n-- Creating Directories --")
    create_directories()
    
    # Check Python version
    print("\n-- Checking Python Version --")
    if not check_python_version():
        return 1
    
    # Check if running on Raspberry Pi
    print("\n-- Checking System --")
    check_raspberry_pi()
    
    # Check system packages first
    print("\n-- Checking System Packages --")
    check_system_packages()
    
    # Check camera module
    print("\n-- Checking Camera Module --")
    check_camera_module()
    
    # Install requirements (skip if requested)
    if not args.skip_pip:
        print("\n-- Installing Requirements --")
        print("(Use --skip-pip flag to skip this step and use system packages instead)")
        install_requirements(skip_pip=False)
    else:
        print("\n-- Skipping Pip Installation --")
        print("Using system packages as requested")
    
    print("\n=== Setup Complete ===")
    print("Run 'python system_check.py' to verify the installation.")
    print("Run 'python app.py' to start the application.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 