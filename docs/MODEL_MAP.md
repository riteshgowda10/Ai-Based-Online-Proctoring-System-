# EXAMGUARD AI — ML MODEL & VISION DETECTOR SPECIFICATION

This document details all machine learning models, computer vision algorithms, inputs, outputs, thresholds, and performance targets utilized by **EXAMGUARD AI**.

---

## 1. Detector & Model Registry

| Model / Detector | Framework & Source | Weights File / Engine | Input Specs | Output Signal | Target Confidence / Threshold | Execution FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face Detector** | PyTorch / `facenet-pytorch` | `MTCNN` network weights | RGB Frame (640x480 or 1280x720) | Face Bounding Boxes, Detection Probabilities | Confidence $\ge 0.80$, Min Face Size: 40px | ~15–30 FPS |
| **Multi-Face Detector** | PyTorch / `facenet-pytorch` | `MTCNN` network weights | RGB Frame | Face Count, Secondary Face Bounding Boxes | Confidence $> 0.90$, $\ge 2$ faces | ~15–30 FPS |
| **Gaze & Eye Tracker** | MediaPipe FaceMesh | 468 3D Landmark Mesh | RGB Frame | Eye Aspect Ratio (EAR), Horizontal Gaze Offset (Left, Right, Center) | Gaze Threshold: $\pm 15$ px, EAR Blink: 0.3 | ~30 FPS |
| **Head Pose Tracker** | MediaPipe FaceMesh & OpenCV `solvePnP` | 3D Generic Facial Landmark Model | 6 key 3D Facial Landmarks (Nose, Chin, Eye Corners, Mouth Corners) | 3D Euler Angles (Yaw, Pitch, Roll in degrees) | Yaw: $\pm 25^\circ$, Pitch Down: $> 18^\circ$ (Phone Risk) | ~30 FPS |
| **Mouth Activity Monitor** | MediaPipe FaceMesh | Lip Landmark Vertices (Indices 13, 14, 78, 306) | RGB Frame | Mouth Openness Ratio, Mouth Width | Openness $> 0.03$ or Width $> 0.20$ | ~30 FPS |
| **Object Detector** | Ultralytics YOLOv8 | `models/yolov8n.pt` (6.2 MB) | Resized RGB Frame ($320 \times 320$) | Bounding Box, Class ID (`cell phone`, `book`, `laptop`) | Confidence $\ge 0.65$, Max FPS: 5 FPS | ~10–25 FPS |
| **Audio Voice Activity** | Web Audio API / PyAudio | Energy & Zero Crossing Rate (ZCR) engine | 16kHz PCM Audio Chunks (512 samples) | Voice Activity Boolean, Energy Level | Energy $> 0.001$, ZCR $< 0.35$ | Real-time (Low Latency) |

---

## 2. Standardized Detector Architecture Interface

All detector classes in EXAMGUARD AI conform conceptually to a single unified base interface:

```python
class BaseDetector:
    def initialize(self):
        """Initialize models, load weights onto CUDA/CPU."""
        pass

    def process(self, frame, timestamp=None):
        """Process incoming video/audio frame and return normalized DetectionResult."""
        pass

    def get_result(self):
        """Return the latest cached detection state."""
        pass

    def shutdown(self):
        """Release GPU/CPU resources cleanly."""
        pass
```

---

## 3. High-Precision Temporal Validation Rules

To prevent false alarms caused by transient single-frame glitches or quick natural blinks, EXAMGUARD AI enforces strict temporal debouncing:

1. **Face Disappearance**: Requires absence for $> 5.0$ consecutive seconds before emitting a `FACE_DISAPPEARED` event.
2. **Multiple Faces**: Requires $\ge 2$ high-confidence faces ($p > 0.90$) for $\ge 5$ consecutive frames before emitting a `MULTIPLE_FACES` event.
3. **Off-Screen Phone / Head Pose Risk**: Requires looking down Pitch $> 18^\circ$ or Yaw $> 25^\circ$ for $> 2.0$ seconds before triggering a `HEAD_POSE_DEVIATION` alert.
4. **Forbidden Objects**: YOLO detections are processed at 5 FPS max with a 10-second alert cooldown to avoid notification spam.
