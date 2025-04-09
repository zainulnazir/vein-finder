/*
 * Vein Finder - IR LED Controller
 * 
 * This sketch controls 8 IR LEDs in a ring formation
 * for vein imaging applications.
 */

// LED pins (adjust as needed based on your wiring)
const int ledPins[] = {2, 3, 4, 5, 6, 7, 8, 9};
const int numLeds = 8;

// Default settings
int brightness = 255;  // 0-255
int pattern = 1;       // 1: all on, 2: alternate, 3: sequential
bool ledsOn = true;

// For sequential pattern timing
unsigned long lastChangeTime = 0;
const int sequentialDelay = 100;  // ms
int currentLed = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize LED pins as outputs
  for (int i = 0; i < numLeds; i++) {
    pinMode(ledPins[i], OUTPUT);
  }
  
  // Initial state - all LEDs on
  updateLEDs();
  
  Serial.println("Vein Finder LED Controller ready");
}

void loop() {
  // Check for serial commands
  checkSerialCommands();
  
  // Update LED pattern if sequential and LEDs are on
  if (pattern == 3 && ledsOn) {
    updateSequentialPattern();
  }
}

void checkSerialCommands() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == 'B') {  // Brightness command
      int newBrightness = Serial.parseInt();
      if (newBrightness >= 0 && newBrightness <= 255) {
        brightness = newBrightness;
        updateLEDs();
        Serial.println("OK");
      } else {
        Serial.println("ERR");
      }
    }
    else if (command == 'P') {  // Pattern command
      int newPattern = Serial.parseInt();
      if (newPattern >= 1 && newPattern <= 3) {
        pattern = newPattern;
        updateLEDs();
        Serial.println("OK");
      } else {
        Serial.println("ERR");
      }
    }
    else if (command == 'T') {  // Toggle command
      int state = Serial.parseInt();
      ledsOn = (state == 1);
      updateLEDs();
      Serial.println("OK");
    }
    
    // Consume any remaining input
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}

void updateLEDs() {
  if (!ledsOn) {
    // Turn all LEDs off
    for (int i = 0; i < numLeds; i++) {
      analogWrite(ledPins[i], 0);
    }
    return;
  }
  
  switch (pattern) {
    case 1:  // All LEDs on
      for (int i = 0; i < numLeds; i++) {
        analogWrite(ledPins[i], brightness);
      }
      break;
      
    case 2:  // Alternate LEDs
      for (int i = 0; i < numLeds; i++) {
        if (i % 2 == 0) {
          analogWrite(ledPins[i], brightness);
        } else {
          analogWrite(ledPins[i], 0);
        }
      }
      break;
      
    case 3:  // Sequential (handled in loop)
      // Reset all LEDs
      for (int i = 0; i < numLeds; i++) {
        analogWrite(ledPins[i], 0);
      }
      // Turn on current LED
      analogWrite(ledPins[currentLed], brightness);
      break;
  }
}

void updateSequentialPattern() {
  // Check if it's time to update the sequential pattern
  unsigned long currentTime = millis();
  if (currentTime - lastChangeTime >= sequentialDelay) {
    // Turn off current LED
    analogWrite(ledPins[currentLed], 0);
    
    // Move to next LED
    currentLed = (currentLed + 1) % numLeds;
    
    // Turn on new LED
    analogWrite(ledPins[currentLed], brightness);
    
    // Update time
    lastChangeTime = currentTime;
  }
} 