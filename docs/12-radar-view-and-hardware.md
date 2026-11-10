# 12. The radar view and the hardware trigger

This document explains the real-time Radar HUD visualization and microcontroller-based hardware trigger system.

> **Status.** The radar classes (`src/inference/radar.py`) were run headless on validation images and are covered by unit tests; the figure in the README is their output. The live window of `scripts/radar_detect.py` and webcam input were not opened in testing, and the Arduino part is untested.

## 1. Tactical Radar HUD Display

The radar HUD (`scripts/radar_detect.py`) transforms standard optical camera frames into a tactical split-screen interface displaying:
1. **Left Panel**: Annotated camera feed with color-coded bounding boxes, confidence tags, a detection count and a running-average FPS counter.
2. **Right Panel**: A circular scope with a target blip at the position of each detection's box centre in the frame. It is a linear map of image coordinates, not range or bearing.

```
+------------------------------------+------------------------------------+
|                                    |                 N                  |
|                                    |                 |                  |
|           OPTICAL FEED             |        W --- (  +  ) --- E         |
|      [ Aircrafts / Bird / Drone ]  |                 |                  |
|                                    |                 S                  |
|      Detections: 2 | FPS: xx.x     |   TACTICAL RADAR HUD | TRACKS: 2   |
+------------------------------------+------------------------------------+
```

### Coordinate Transformation

Bounding box centers in camera pixel space `(cx, cy)` are mapped onto the scope using normalized offsets:

$$\Delta x = \frac{c_x - W/2}{W/2}, \quad \Delta y = \frac{c_y - H/2}{H/2}$$

Radar coordinates:
$$X_{\text{radar}} = C_x + \Delta x \cdot (R - 10), \quad Y_{\text{radar}} = C_y + \Delta y \cdot (R - 10)$$

where $C$ is the scope centre and $R$ its radius in pixels (`src/inference/radar.py`). Points that fall outside the circle are pulled back onto its edge.

### Launching the Radar Interface

```bash
# Live webcam (index 0)
python scripts/radar_detect.py --model weights/best.pt --source 0

# Video file
python scripts/radar_detect.py --model weights/best.pt --source path/to/your_video.mp4

# Save the combined output as a video
python scripts/radar_detect.py --model weights/best.pt --source path/to/your_video.mp4 --save radar_demo.mp4
```

---

## 2. Arduino Ultrasonic Sensor Trigger

> **Status: implemented, not verified.** The listener and the sketch below have not been run against real hardware.

The hardware listener (`scripts/hardware_trigger.py`) enables autonomous perimeter monitoring. An Arduino connected to an ultrasonic distance sensor (such as HC-SR04) continuously scans for approaching objects. When an object enters proximity, the microcontroller broadcasts a trigger signal over serial.

### Serial Protocol

- **Baud Rate**: 9600 bps
- **Trigger Keyword**: `DETECT`

### Arduino Example Sketch

```cpp
const int trigPin = 9;
const int echoPin = 10;
const int thresholdDistanceCm = 150; // Trigger distance

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH);
  long distanceCm = duration * 0.034 / 2;

  if (distanceCm > 0 && distanceCm < thresholdDistanceCm) {
    Serial.println("DETECT");
    delay(5000); // Cooldown delay
  }
  delay(100);
}
```

### Starting the Listener

```bash
python scripts/hardware_trigger.py --port COM3 --baud 9600 --trigger-cmd scripts/radar_detect.py
```

Back to the [guide index](../README.md).
