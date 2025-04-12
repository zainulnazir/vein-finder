#!/usr/bin/env python3

import serial
import time
import os


class LEDController:
    def __init__(self, port=None, baud_rate=9600, timeout=1, simulation=False):
        """Initialize connection to Arduino for LED control

        Args:
            port: Serial port name, None for auto-detection
            baud_rate: Serial baud rate
            timeout: Serial timeout in seconds
            simulation: Use simulation mode (no hardware required)
        """
        self.connected = False
        self.ser = None
        self.simulation = simulation
        self.simulated_led_state = False
        self.simulated_brightness = 255
        self.simulated_pattern = 1

        # If in simulation mode, just return
        if self.simulation:
            print("LED Controller running in simulation mode")
            self.connected = True
            return

        # Try to find Arduino if port not specified
        if port is None:
            port = self._find_arduino_port()

        # If still no port found, use default
        if port is None:
            # Try common Arduino port names
            potential_ports = [
                "/dev/ttyACM0",
                "/dev/ttyACM1",
                "/dev/ttyUSB0",
                "/dev/ttyUSB1",
                # Add Mac/Windows common ports
                "/dev/cu.usbmodem*",
                "/dev/cu.usbserial*",
                "COM3",
                "COM4",
            ]

            # Try each port with a short timeout
            for p in potential_ports:
                # Handle wildcard paths
                if "*" in p and os.path.exists("/dev"):
                    import glob

                    matching_ports = glob.glob(p)
                    for match in matching_ports:
                        if self._try_connect(
                            match, baud_rate, 0.5
                        ):  # Short timeout for check
                            port = match
                            break
                elif os.path.exists(p):
                    if self._try_connect(p, baud_rate, 0.5):  # Short timeout for check
                        port = p
                        break

        # If port identified, make proper connection with full timeout
        if port:
            self._try_connect(port, baud_rate, timeout)

    def _try_connect(self, port, baud_rate, timeout):
        """Try to connect to a port with specified parameters"""
        try:
            # Close existing connection if any
            if self.ser and self.ser.is_open:
                self.ser.close()

            # Try to connect with short timeout to avoid blocking
            self.ser = serial.Serial(port, baud_rate, timeout=timeout)

            # Send a test command and check response
            self.ser.write("Q\n".encode())
            response = self.ser.readline()

            if response:  # Got any response at all
                self.connected = True
                print(f"Connected to Arduino on {port}")
                return True
            else:
                self.ser.close()
                return False

        except Exception as e:
            print(f"Failed to connect on {port}: {e}")
            if self.ser and self.ser.is_open:
                self.ser.close()
            return False

    def _find_arduino_port(self):
        """Try to automatically find Arduino port"""
        try:
            import serial.tools.list_ports

            ports = list(serial.tools.list_ports.comports())

            # First look for Arduino in description
            for port in ports:
                if "Arduino" in port.description:
                    return port.device

            # Then try common identifiers
            for port in ports:
                if any(
                    id in port.device for id in ["ACM", "USB", "usbmodem", "usbserial"]
                ):
                    return port.device

            # If we get here, no Arduino port found
            if ports:
                print(f"Available ports: {', '.join([p.device for p in ports])}")
            else:
                print("No serial ports found")

            return None

        except ImportError:
            print("pyserial-tools not available, cannot auto-detect Arduino")
            return None

    def __del__(self):
        """Clean up serial connection when object is destroyed"""
        if hasattr(self, "ser") and self.ser.is_open:
            self.ser.close()

    def set_brightness(self, brightness):
        """Set LED brightness (0-255)"""
        if not self.connected:
            return False

        if self.simulation:
            self.simulated_brightness = brightness
            print(f"Simulation: LED brightness set to {brightness}")
            return True

        # Send command to Arduino (format: 'B<brightness>')
        command = f"B{brightness}\n"
        self.ser.write(command.encode())

        # Wait for acknowledgement
        response = self.ser.readline().decode().strip()
        return response == "OK"

    def set_pattern(self, pattern):
        """Set LED pattern (1: all on, 2: alternate, 3: sequential)"""
        if not self.connected:
            return False

        if self.simulation:
            self.simulated_pattern = pattern
            print(f"Simulation: LED pattern set to {pattern}")
            return True

        # Send command to Arduino (format: 'P<pattern>')
        command = f"P{pattern}\n"
        self.ser.write(command.encode())

        # Wait for acknowledgement
        response = self.ser.readline().decode().strip()
        return response == "OK"

    def toggle_leds(self, state=None):
        """Turn LEDs on or off (True: on, False: off) or toggle if state is None"""
        if not self.connected:
            return False

        if self.simulation:
            # Handle simulated LED state
            if state is None:
                self.simulated_led_state = not self.simulated_led_state
            else:
                self.simulated_led_state = state
            print(
                f"Simulation: LEDs turned {'ON' if self.simulated_led_state else 'OFF'}"
            )
            return self.simulated_led_state

        # If state is None, toggle current state
        if state is None:
            # First query current state
            self.ser.write("Q\n".encode())
            response = self.ser.readline().decode().strip()
            try:
                current_state = response == "ON"
                return self.toggle_leds(not current_state)  # Toggle the state
            except:
                # If query fails, default to turning on
                return self.toggle_leds(True)

        # Send command to Arduino (format: 'T<state>')
        command = f"T{1 if state else 0}\n"
        self.ser.write(command.encode())

        # Wait for acknowledgement
        response = self.ser.readline().decode().strip()
        return state if response == "OK" else False
