# Vein Finder

A medical imaging device for visualizing veins to assist with venipuncture procedures.

## Project Overview

This project creates a vein imaging system using near-infrared (NIR) light to improve the visualization of veins for medical procedures. The system uses IR LEDs (850nm) to illuminate the skin, a NoIR camera to capture the images, and advanced image processing techniques to enhance the visibility of veins.

### Hardware Components

- Raspberry Pi 4B
- Raspberry Pi NoIR Camera v2
- 8 IR LEDs (850 nm) arranged in a ring
- Arduino Uno (for LED control)
- Physical IR filter (to block visible light)

### Key Features

- Live video streaming with real-time vein detection
- Multiple vein detection algorithms:
  - Adaptive thresholding
  - Frangi filter (tubular structure enhancement)
  - Laplacian edge detection
- Contrast enhancement options:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Histogram equalization
- LED control:
  - Brightness adjustment
  - Multiple lighting patterns
- Image capture and storage
- Web-based user interface

## Installation

### Prerequisites

- Raspberry Pi with Raspberry Pi OS (Bookworm)
- Arduino IDE
- Python 3.9+

### Hardware Setup

1. Connect the NoIR Camera to the Raspberry Pi CSI port
2. Connect Arduino Uno to Raspberry Pi via USB
3. Wire the IR LEDs to Arduino pins 2-9 with appropriate resistors
4. Position the IR filter in front of the camera lens

### Software Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/vein-finder.git
   cd vein-finder
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Upload the Arduino sketch:
   - Open `arduino_sketch/vein_finder_led_control.ino` in Arduino IDE
   - Connect Arduino to your computer
   - Select the appropriate board and port
   - Upload the sketch

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Access the web interface:
   - Open a browser on the Raspberry Pi or any device on the same network
   - Navigate to `http://<raspberry-pi-ip>:8000`

3. Adjust settings for optimal vein visualization:
   - Try different detection methods for various skin types and lighting conditions
   - Adjust exposure and gain based on ambient light
   - Control LED brightness for best contrast

4. Capture and save images for documentation or further analysis

## Notes for Medical Use

This device is intended as an educational or assistive tool and not as a primary medical diagnostic device. Always follow proper clinical protocols when performing venipuncture procedures.

For best results:
- Use in a dimly lit room to reduce interference from visible light
- Position the device 10-15 cm from the skin surface
- Keep the device steady during imaging
- Try different vein detection methods for different patients

## Troubleshooting

- **Arduino not detected**: Check USB connection and verify COM port
- **Poor image quality**: Adjust camera exposure and gain settings
- **LEDs not functioning**: Check Arduino connections and verify the sketch has uploaded correctly
- **Veins not visible**: Try a different detection method or adjust contrast settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was created as a college project
- Thanks to OpenCV and scikit-image for image processing libraries 