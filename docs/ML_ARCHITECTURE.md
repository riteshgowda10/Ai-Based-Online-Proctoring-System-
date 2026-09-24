# EXAMGUARD AI — MODULAR ML & COMPUTER VISION ARCHITECTURE

## 1. Architectural Pipeline Flow

```
                      +-------------------+
                      |   WEBCAM STREAM   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   Frame Capture   |
                      | & Normalization   |
                      +---------+---------+
                                |
       +------------------------+------------------------+
       |                        |                        |
       v                        v                        v
+--------------+         +--------------+         +--------------+
|  Face & Pose |         |    Object    |         |   Identity   |
|  Landmarks   |         |  Detection   |         | Verification |
| (MTCNN+Mesh) |         |   (YOLOv8)   |         |  (FaceNet)   |
+-------+------+         +-------+------+         +-------+------+
        |                        |                        |
        v                        |                        |
  3D Head Pose                   |                        |
  & Gaze Vectors                 |                        |
        |                        |                        |
        +------------------------+------------------------+
                                 |
                                 v
               +----------------------------------+
               |     Temporal Behavior Engine     |
               | (Sliding Window & Debouncing)    |
               +-----------------+----------------+
                                 |
                                 v
               +----------------------------------+
               |     Proctoring Risk Engine       |
               |  (Configurable Weight Scoring)   |
               +-----------------+----------------+
                                 |
                                 v
               +----------------------------------+
               |     REST & Telemetry API         |
               +-----------------+----------------+
                                 |
                                 v
               +----------------------------------+
               |    Admin & Proctor Dashboard     |
               +----------------------------------+
```

## 2. Component Design Principles
1. **Separation of Concerns**: Visual detectors focus solely on spatial frame-level inference. Temporal logic (saccade filtering, disappearance timers, multiple-face debouncing) is isolated in the Temporal Event Engine.
2. **Deterministic Explainability**: Proctoring decisions produce human-readable event breakdowns (`contributors`) with timestamps and durations rather than opaque scalar classifications.
3. **Hardware Agnostic Runtime**: Primary CPU support with transparent CUDA GPU auto-negotiation.
