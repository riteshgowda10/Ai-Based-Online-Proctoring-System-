# EXAMGUARD AI — DEPENDENCY & ENVIRONMENT MAP

This document maps all core Python dependencies, system libraries, optional modules, and platform compatibility fallbacks for **EXAMGUARD AI**.

---

## 1. Core Dependency Inventory

| Package Name | Minimum Version | Category | Purpose | Fallback / Handling Strategy |
| :--- | :--- | :--- | :--- | :--- |
| `flask` | $\ge 2.0.0$ | Web Framework | Routing, Jinja2 template rendering, HTTP API endpoints, MJPEG streaming. | Mandatory core dependency. |
| `opencv-python` | $\ge 4.5.0$ | Computer Vision | Video capture, frame preprocessing, 3D Head Pose PnP math, color space conversion. | Mandatory core dependency. |
| `facenet-pytorch` | $\ge 2.5.0$ | Deep Learning | MTCNN face detection network for single/multi-face counts. | Mandatory core dependency. |
| `mediapipe` | $< 1.0.0$ (0.10.35) | Machine Learning | 468 landmark FaceMesh for eye tracking, EAR, and mouth movement. | Wrapped with `try/except AttributeError` for version compatibility. |
| `ultralytics` | $\ge 8.0.0$ | Object Detection | YOLOv8n object detection engine for mobile phones, books, and laptops. | Automatically downloads `yolov8n.pt` to `models/` directory if missing. |
| `torch` / `torchvision` | $\ge 1.7.0$ | ML Framework | Tensor computations and CUDA/CPU hardware acceleration for MTCNN & YOLO. | CPU fallback enabled automatically if CUDA is unavailable. |
| `numpy` | $\ge 1.20.0$ | Math Utility | Vector matrix calculations for EAR, head pose vectors, and image transformations. | Mandatory core dependency. |
| `pyyaml` | $\ge 5.0.0$ | Configuration | Parses `config/config.yaml`. | Mandatory core dependency. |
| `mss` | $\ge 6.1.0$ | Screen Capture | Multi-monitor high-speed screen capture utility. | Optional / Fallback to browser Web API screen events. |
| `gTTS` / `pygame-ce` | $\ge 2.3.1$ / $\ge 2.1.2$ | Audio Alert Synthesis | Generates voice alert audio files and plays back sound effects. | Wrapped in `try/except` to prevent audio driver crashes. |
| `pdfkit` / `jinja2` / `matplotlib` | $\ge 1.0.0$ / $\ge 3.0.0$ / $\ge 3.5.0$ | Reporting | Generates PDF / HTML proctoring reports with timeline charts & frequency heatmaps. | Automatic fallback to HTML format if `wkhtmltopdf` executable is missing. |
| `pyaudio` | $\ge 0.2.13$ | Audio Capture | Real-time microphone audio stream sampling. | Optional fallback to browser Web Audio API in candidate UI if C++ build tools are missing on Windows. |

---

## 2. Environment Variables & Paths

- `KMP_DUPLICATE_LIB_OK=TRUE`: Prevents OpenMP library duplicate initialization errors when PyTorch, OpenCV, and MediaPipe execute concurrently on Windows.
- `PYTHONPATH`: Configured to include project root `src/` directory.
- `CONFIG_PATH`: Relative configuration path pointing to `config/config.yaml`.

---

## 3. Platform Verification Command (Windows PowerShell)

```powershell
# Environment Verification Command
C:\Users\roope\anaconda3\python.exe -c "import cv2, torch, facenet_pytorch, mediapipe, ultralytics, flask, yaml; print('Environment Validation: ALL MODULES LOADED OK')"
```
