# AI-Based Online Examination Proctoring System (PRJ 137)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, privacy-aware **AI-Based Online Proctoring System** engineered for computerized academic assessments. The platform performs real-time continuous candidate identity verification, multi-person detection, prohibited device detection, 3D head-pose tracking, eye-gaze estimation, and temporal behavioral anomaly analysis.

**Author**: Ritesh A S  
**GitHub**: [https://github.com/riteshgowda10](https://github.com/riteshgowda10)  

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
  - [Frontend](#frontend)
  - [Backend](#backend)
  - [Machine Learning & Vision](#machine-learning--vision)
  - [Admin Control Center](#admin-control-center)
  - [Candidate System](#candidate-system)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [ML Setup](#ml-setup)
  - [Database Setup](#database-setup)
- [Environment Variables](#environment-variables)
- [Model Setup](#model-setup)
- [Dataset Setup](#dataset-setup)
- [Running the System](#running-the-system)
- [Training ML Models](#training-ml-models)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Admin Features](#admin-features)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [GitHub Setup](#github-setup)

---

## Overview
Online examinations require rigorous academic integrity without invasive, opaque surveillance. ExamGuard AI addresses this with an explainable multi-signal vision pipeline that never issues automated disqualifications. Instead, the system computes an auditable **Candidate Integrity Index** ($0 - 100$) and generates structured temporal evidence logs for human proctor review.

---

## Features

### Frontend
- **Hyper-Neon Obsidian UI**: Modern dark theme (`bg-[#070b14]`) built with Tailwind CSS and FontAwesome 6 icons.
- **Biometric Candidate Onboarding**: On-screen webcam alignment and identity verification pre-check.
- **Computer-Based Testing (CBT) Portal**: Split-pane interface featuring coding problems, Monaco-style test case runner, live timer, and webcam feed.
- **Client-Side Anti-Cheating Defenses**: Intercepts tab-switching (`blur`/`visibilitychange`), clipboard copying (`Ctrl+C`/`Ctrl+V`), and screenshot captures (`PrtScn`/Snipping tool).

### Backend
- **Flask 3.1 WSGI Framework**: Modular RESTful backend with session state tracking, CORS negotiation, and JSON error handling.
- **Real-Time MJPEG Streaming**: Efficient multi-threaded camera ingestion with 3D landmark overlays.
- **Standardized Proctoring APIs**: Dedicated endpoints for starting, stopping, frame inference, and event retrieval.
- **Audit Trail Logger**: Append-only transaction logging with configurable debouncing.

### Machine Learning & Vision
- **Object Detection (YOLOv8)**: Detects persons, cell phones, laptops, books, and earphones.
- **Biometric Identity Verification (FaceNet)**: Uses pre-trained `InceptionResnetV1` (VGGFace2) to extract 512-D embeddings and verify candidate identity using Cosine Similarity ($\ge 0.65$).
- **Single-Pass Face Detection (MTCNN)**: Fast detection of primary candidate and secondary unauthorized persons.
- **3D Head-Pose Estimation**: OpenCV `solvePnP` computes Euler angles (Yaw, Pitch, Roll) with temporal moving-average smoothing.
- **Eye-Gaze Tracking**: Normalized pupil-to-nose geometry detects looking left, right, up, down, or center.
- **Temporal Event Engine**: Debounces transient blips (4s disappearance, 5 consecutive frames for multiple faces, 8s alert cooldown).
- **Proctoring Risk Engine**: Weighted formulation computing explainable integrity scores and risk categories (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`).

### Admin Control Center
- **Live Video Matrix**: Grid monitoring candidate camera feeds with latency indicators.
- **Live Candidate Malpractice Modal**: Real-time breakdown of copy attempts, screenshot interceptions, tab switches, warning flags, and active ML model inferences.
- **One-Click Session Controls**: Proctor broadcast announcements, pause session, terminate session, and generate PDF malpractice audit reports.

### Candidate System
- **Pre-Exam Environment Verification**: Checks lighting, webcam availability, and browser permissions.
- **Dual Authentication**: Username/password credentials paired with biometric facial enrollment.

---

## Architecture

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
               +-----------------+----------------+
```

---

## Technology Stack
- **Languages**: Python 3.10+, JavaScript (ES6+), HTML5, CSS3, SQL
- **Computer Vision & Deep Learning**: PyTorch 2.13, Ultralytics YOLOv8, `facenet-pytorch`, MediaPipe, OpenCV
- **Machine Learning & Analytics**: Scikit-Learn 1.9, SciPy, Pandas, NumPy
- **Backend & Networking**: Flask 3.1, Jinja2, PyAudio, Requests, Werkzeug
- **Reporting**: ReportLab, pdfkit, Matplotlib
- **Database**: SQLite3 / PostgreSQL

---

## Project Structure
```
AI-Based-Online-Proctoring-System/
├── .env.example                       # Environment template
├── .gitignore                          # Git exclusions
├── LICENSE                             # MIT License
├── README.md                           # Master project documentation
├── requirements.txt                    # Python dependencies
│
├── configs/                            # System configurations
│   ├── training.yaml                   # Model training hyperparameters
│   ├── inference.yaml                  # Real-time inference settings
│   └── proctoring.yaml                 # Event weights & risk thresholds
│
├── config/                             # Legacy compatibility config
│   └── config.yaml                     # Application settings
│
├── database/                           # Database code & migration
│   ├── schema.sql                      # Relational table schema
│   └── README.md                       # Database setup guide
│
├── dataset/                            # Proctoring calibration dataset
│   ├── images/ (train, val, test)      # Verification frames
│   ├── labels/ (train, val, test)      # YOLO normalized annotations
│   └── build_proctoring_dataset.py     # Dataset builder
│
├── datasets/                           # Dataset documentation
│   └── README.md                       # Dataset specifications & acquisition
│
├── docs/                               # Engineering documentation
│   ├── ML_MODEL_AUDIT.md               # Pre-development audit matrix
│   ├── ML_ARCHITECTURE.md              # System design & data flow
│   ├── DATASET_GUIDE.md                # Annotation & privacy guide
│   ├── TRAINING_GUIDE.md               # Training instructions
│   ├── EVALUATION.md                   # Real benchmark figures
│   ├── INFERENCE_GUIDE.md              # REST & Python inference guide
│   ├── MODEL_LIMITATIONS.md            # Edge case analysis
│   ├── GITHUB_PROJECT_AUDIT.md         # Pre-commit file audit
│   └── FINAL_ML_REPORT.md              # Complete execution report
│
├── evaluation/                         # Evaluation artifacts
│   └── object_detection/
│       ├── metrics.json                # JSON benchmark output
│       ├── metrics.csv                 # Tabular per-class metrics
│       └── predictions/                # Annotated test visuals
│
├── ml/                                 # Modular Machine Learning package
│   ├── object_detection/               # YOLO training, validation, inference
│   │   ├── dataset.yaml                # YOLO dataset configuration
│   │   ├── validate_dataset.py         # Quality assurance & corruption checks
│   │   ├── train.py                    # Training & fine-tuning loop
│   │   ├── validate.py                 # Benchmark engine
│   │   ├── predict.py                  # Visual prediction renderer
│   │   ├── inference.py                # Real-time detector class
│   │   └── README.md                   # Object detection docs
│   ├── face/                           # Face detection & identity
│   │   ├── detector.py                 # Single-pass MTCNN detector
│   │   └── identity.py                 # FaceNet biometric identity verifier
│   ├── head_pose/                      # 3D Head pose
│   │   └── estimator.py                # solvePnP Euler angle tracker
│   ├── gaze/                           # Eye gaze
│   │   └── estimator.py                # Normalized EAR & saccade tracker
│   ├── behavior/                       # Temporal events & risk
│   │   ├── event_engine.py             # Sliding window debouncer
│   │   └── risk_engine.py              # Configurable risk indicator
│   └── inference/                      # Orchestration
│       └── pipeline.py                 # Unified real-time pipeline
│
├── models/                             # Model weights registry
│   ├── yolov8n.pt                      # Base YOLOv8 checkpoint (6.2 MB)
│   ├── object_detection/best.pt        # Fine-tuned checkpoint (6.2 MB)
│   ├── malpractice_risk_model.pkl      # Random Forest classifier (4.3 MB)
│   ├── head_pose_gaze_model.pkl        # Posture classifier (1.2 MB)
│   ├── candidate_anomaly_detector.pkl  # Isolation Forest model (1.3 MB)
│   ├── README.md                       # Checkpoint guide
│   └── MODEL_REGISTRY.md               # Official model registry
│
├── src/                                # Application source code
│   ├── dashboard/                      # Web application
│   │   ├── app.py                      # Flask WSGI entrypoint
│   │   └── templates/                  # Frontend templates
│   │       ├── admin.html              # Admin Control Center
│   │       ├── exam.html               # Candidate CBT exam interface
│   │       ├── login.html              # Authentication & registration
│   │       ├── instructions.html       # Candidate test briefing
│   │       └── dashboard.html          # Candidate live status
│   ├── detection/                      # Detector interfaces
│   ├── reporting/                      # PDF report generator
│   ├── training/                       # Machine learning training scripts
│   │   ├── dataset_generator.py        # Telemetry dataset generator
│   │   └── train_models.py             # ML training pipeline
│   └── utils/                          # Logging, audio, alert utilities
│
└── tests/                              # Automated test suite
    └── ml/
        ├── test_face_identity.py       # Face & identity tests
        ├── test_head_pose_gaze.py      # Pose & gaze tests
        ├── test_objects_behavior.py    # Object & event tests
        └── test_pipeline.py            # End-to-end integration tests
```

---

## Installation

### Backend Setup
1. **Clone the Repository**:
   ```powershell
   git clone https://github.com/riteshgowda10/AI-Based-Online-Proctoring-System.git
   cd AI-Based-Online-Proctoring-System
   ```
2. **Install Python Dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Frontend Setup
The frontend uses CDN-delivered Tailwind CSS and FontAwesome. No separate node build step is required. Templates are served directly by Flask from `src/dashboard/templates/`.

### ML Setup
All model checkpoints (`models/*.pt`, `models/*.pkl`) are pre-packaged. To re-download or initialize:
```powershell
python -m ml.object_detection.validate_dataset
```

### Database Setup
Initialize the SQLite schema:
```powershell
# Using Python
python -c "import sqlite3; conn = sqlite3.connect('database/proctoring.db'); conn.executescript(open('database/schema.sql').read()); conn.close()"
```

---

## Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Default configuration works out-of-the-box for local testing.

---

## Running the System

Start the Flask server:
```powershell
python src/dashboard/app.py
```

Access portals in your browser:
- **Candidate Login / Registration**: [http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)
- **Candidate CBT Exam Interface**: [http://127.0.0.1:5000/exam](http://127.0.0.1:5000/exam)
- **Admin Control Center**: [http://127.0.0.1:5000/admin](http://127.0.0.1:5000/admin)

Default Credentials:
- **Admin**: Username `admin` / Password `admin123`
- **Candidate**: Username `STUDENT_001` / Password `123456`

---

## Training ML Models

### 1. Train Multi-Modal Proctoring Models
```powershell
python src/training/train_models.py
```
Outputs:
- `models/malpractice_risk_model.pkl` (Accuracy: 95.58%)
- `models/head_pose_gaze_model.pkl` (Accuracy: 99.90%)
- `models/candidate_anomaly_detector.pkl` (ROC-AUC: 1.0000)

### 2. Fine-Tune YOLOv8 Object Detector
```powershell
python -m ml.object_detection.train
```

### 3. Evaluate & Profile
```powershell
python -m ml.object_detection.validate
```

---

## Testing
Run the complete automated unit test suite:
```powershell
python -m unittest discover -s tests\ml -p "test_*.py" -v
```
All **11 unit tests** pass with 100% success rate.

---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `POST /api/proctoring/start` | POST | Starts proctoring session |
| `POST /api/proctoring/stop` | POST | Stops proctoring session |
| `POST /api/proctoring/frame` | POST | Analyzes webcam frame and returns events & risk score |
| `GET /api/proctoring/events` | GET | Returns event history for the session |
| `GET /api/proctoring/status` | GET | Returns current integrity index and risk level |
| `GET /api/stats` | GET | Returns real-time telemetry stream |
| `GET /api/admin/candidate_report/<student_id>` | GET | Returns full malpractice audit report data |
| `POST /api/generate_report` | POST | Generates downloadable PDF/HTML report |

---

## Limitations
- **Lighting Sensitivity**: Backlighting can degrade facial landmark visibility.
- **Eyeglass Reflections**: Glare on lenses can momentarily alter Eye Aspect Ratio.
- **Single Camera View**: Rear desk area is not visible without an auxiliary smartphone camera.

---

## Future Improvements
- TensorRT / ONNX Runtime export for 60+ FPS multi-stream processing.
- Multi-camera integration pairing a side-phone view with the primary webcam.
- WebAssembly (Wasm) in-browser client-side inference to eliminate server video streaming overhead.

---

## GitHub Setup
To link and push this complete system to your repository:
```powershell
git init
git add .
git commit -m "feat: complete AI-based online proctoring system with frontend backend ML and docs"
git branch -M main
git remote add origin https://github.com/riteshgowda10/<YOUR_REPOSITORY_NAME>.git
git push -u origin main
```
