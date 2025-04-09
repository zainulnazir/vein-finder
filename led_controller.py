#!/usr/bin/env python3

import serial
import time
import os

class LEDController:
    def __init__(self, port=None, baud_rate=9600, timeout=2):
        """Initialize connection to Arduino for LED control"""
        self.connected = False
        
        # Try to find Arduino if port not specified
        if port is None:
            port = self._find_arduino_port()
            
        # If still no port found, use default
        if port is None:
            # Try common Arduino port names
            potential_ports = [
                '/dev/ttyACM0',
                '/dev/ttyACM1',
                '/dev/ttyUSB0',
                '/dev/ttyUSB1',
                # Add Mac/Windows common ports
                '/dev/cu.usbmodem*',
                '/dev/cu.usbserial*',
                'COM3',
                'COM4'
            ]
            
            # Try each port
            for p in potential_ports:
                # Handle wildcard paths
                if '*' in p and os.path.exists('/dev'):
                    import glob
                    matching_ports = glob.glob(p)
                    if matching_ports:
                        port = matching_ports[0]
                        break
                elif os.path.exists(p):
                    port = p
                    break
            
            # If still no port, use default
            if port is None:
                port = '/dev/ttyACM0'  # Default fallback
                
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=timeout)
            # Allow time for Arduino to reset
            time.sleep(2)
            self.connected = True
            print(f"Connected to Arduino on {port}")
        except Exception as e:
            self.connected = False
            print(f"Failed to connect to Arduino: {e}")
            
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
                if any(id in port.device for id in ["ACM", "USB", "usbmodem", "usbserial"]):
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
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            
    def set_brightness(self, brightness):
        """Set LED brightness (0-255)"""
        if not self.connected:
            return False
            
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
        
        # If state is None, toggle current state
        if state is None:
            # First query current state
            self.ser.write("Q\n".encode())
            response = self.ser.readline().decode().strip()
            try:
                current_state = (response == "ON")
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