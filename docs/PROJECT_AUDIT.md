# EXAMGUARD AI — PHASE 0: COMPLETE REPOSITORY AUDIT

**System Name**: EXAMGUARD AI (AI-Based Online Examination and Intelligent Proctoring Platform)  
**Project Path**: `C:\Users\roope\.gemini\antigravity\scratch\online-exam-proctor`  
**Audit Date**: September 2026  
**Auditor**: Lead System Architect & Senior Engineering Team  

---

## 1. Executive Summary

This document presents a comprehensive, non-destructive audit of the existing codebase. The platform is currently configured as a Flask-based AI proctoring prototype (**PRJ 137 / ExamGuard**). All working modules—including 3D Head Pose tracking, PyTorch MTCNN face detection, YOLOv8n object detection, MediaPipe FaceMesh eye/mouth tracking, Web Audio API mic monitoring, Flask video streaming, anti-screenshot keyboard interception, dynamic candidate risk scoring, and HTML/PDF report generation—have been inspected, validated, and classified.

---

## 2. File Inventory & Classification

Each file in the repository has been evaluated against the architecture specification and assigned an operational action code:
- **[KEEP]**: Functional, stable core module. Retain as-is or extend cleanly.
- **[MODIFY]**: Existing component requiring refactoring, decoupling, or security enhancement.
- **[MERGE]**: Component to be combined with standard registries or models.
- **[NEW]**: Component introduced to fulfill EXAMGUARD AI specification requirements.

| Relative Path | Classification | Purpose & Reuse Strategy |
| :--- | :---: | :--- |
| `src/dashboard/app.py` | **[MODIFY]** | Core Flask web application. Contains route handlers, SSE/polling API endpoints, 3D head pose math, and video stream frame generator. To be refactored into modular blueprints (auth, exam, proctoring, reporting). |
| `src/dashboard/templates/login.html` | **[KEEP/MODIFY]** | Glassmorphism dual login portal (Student & Admin) with candidate registration modal. Form fields and API hooks work cleanly. |
| `src/dashboard/templates/instructions.html` | **[KEEP/MODIFY]** | Pre-exam onboarding setup page with 4-step hardware checks (Camera, Microphone, Fullscreen, Compliance agreement). Fully functional. |
| `src/dashboard/templates/exam.html` | **[KEEP/MODIFY]** | CBT Examination workspace supporting MCQs, multi-language code runner (Python, JS, C++, Java), anti-screenshot blackout defense, and live toast warnings. |
| `src/dashboard/templates/dashboard.html` | **[KEEP/MODIFY]** | Admin & Proctor live monitoring control dashboard with video stream overlay, risk index gauge, telemetry cards, and remote intervention actions. |
| `src/detection/face_detection.py` | **[KEEP]** | PyTorch MTCNN face presence / disappearance tracking. Robust 5-second disappearance debouncing. |
| `src/detection/multi_face.py` | **[KEEP]** | PyTorch MTCNN multi-face count & unauthorized secondary person detection. |
| `src/detection/eye_tracking.py` | **[KEEP]** | MediaPipe FaceMesh Eye Aspect Ratio (EAR) & horizontal gaze deviation tracking with MediaPipe fallback guards. |
| `src/detection/mouth_detection.py` | **[KEEP]** | MediaPipe FaceMesh lip movement / talking monitor with temporal thresholding. |
| `src/detection/object_detection.py` | **[KEEP]** | Ultralytics YOLOv8n object detector for cell phone and book detection. Uses GPU/CPU auto-selection and 320px inference optimization. |
| `src/detection/audio_detection.py` | **[KEEP]** | PyAudio & Web Audio fallback voice activity detector (ZCR & Energy) with optional OpenAI Whisper interface. |
| `src/utils/logging.py` | **[KEEP]** | Alert logger with configurable alert type cooldowns and file logging to `logs/alerts.log`. |
| `src/utils/alert_system.py` | **[KEEP]** | gTTS text-to-speech audio alert synthesizer with pygame-ce playback. |
| `src/utils/violation_logger.py` | **[KEEP]** | JSON violation logger persisting telemetry to `reports/violations.json`. |
| `src/utils/screenshot_utils.py` | **[KEEP]** | Captures violation screenshots and stores images in `reports/violation_captures/`. |
| `src/reporting/report_generator.py` | **[KEEP]** | Generates PDF / HTML candidate proctoring evaluation reports with matplotlib timeline charts and severity heatmaps. |
| `config/config.yaml` | **[MODIFY]** | Central configuration YAML file for video resolutions, thresholds, logging, and model settings. |
| `models/yolov8n.pt` | **[KEEP]** | Pre-trained YOLOv8 nano PyTorch weights file (6.2 MB) for real-time mobile phone and forbidden object detection. |
| `requirements.txt` | **[KEEP]** | Environment dependencies manifest containing PyTorch, OpenCV, MediaPipe, Ultralytics, Flask, etc. |

---

## 3. Working Capabilities Preserved

1. **Face & Multi-Person Detection**: PyTorch MTCNN accurately tracks face count and triggers `FACE_DISAPPEARED` and `MULTIPLE_FACES` alerts.
2. **Gaze & Mouth Tracking**: MediaPipe 468 landmark mesh calculates EAR and gaze offsets.
3. **Object Detection**: YOLOv8n identifies prohibited items (`cell phone`, `book`).
4. **Anti-Screenshot & Snipping Defense**: Keyboard listeners intercept `PrintScreen` and `Win+Shift+S`, applying instant blackout overlays.
5. **CBT & Code Execution Engine**: Interactive MCQ answer tracking and code sandbox compilation for Python, JS, C++, and Java.
6. **Proctor Live Interventions**: Remote warnings, session pause, and session termination endpoints (`/api/proctor/*`).
7. **Report Generation Engine**: Automated Jinja2/HTML and PDF report builder with Matplotlib analytics.

---

## 4. Gaps Identified & Migration Strategy

- **Canonical Route Mapping & Backward Compatibility**: Expose both canonical routes (`/`, `/instructions`, `/exam`, `/dashboard`) and legacy compatibility routes (`/svg`, `/instructionssvg`, `/examsvg`, `/dashboardsvg`).
- **Standard Event Normalization Schema**: Implement a unified `Event` dataclass/JSON schema across all detectors to ensure consistent temporal validation, cooldowns, and risk scoring.
- **Session ID Isolation**: Enforce explicit session tracking (`session_id`) across candidate state transitions, database storage, and proctor WebSocket/SSE streams.
- **ORM Persistence Layer**: Introduce SQLAlchemy models for candidates, proctors, exams, questions, sessions, events, evidence, and interventions while preserving SQLite performance.
