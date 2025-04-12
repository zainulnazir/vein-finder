#!/bin/bash

# Vein Finder Project - Dependency Installer
echo "Installing dependencies for Vein Finder Project..."

# Update package lists
sudo apt-get update

# Install Python dependencies using apt (preferred method)
sudo apt-get install -y python3-pip python3-numpy python3-opencv python3-flask python3-picamera2

# Try to install scikit-image using apt first
if sudo apt-get install -y python3-skimage; then
    echo "Installed scikit-image via apt"
else
    echo "Installing scikit-image via pip with --break-system-packages flag"
    sudo pip3 install scikit-image --break-system-packages
fi

# Try to install flask-cors using apt first
if sudo apt-get install -y python3-flask-cors; then
    echo "Installed flask-cors via apt"
else
    echo "Installing flask-cors via pip with --break-system-packages flag"
    sudo pip3 install flask-cors --break-system-packages
fi

echo "Creating necessary directories..."
mkdir -p static/images

echo "All dependencies installed!"
echo "Run the application with: python3 app.py" 