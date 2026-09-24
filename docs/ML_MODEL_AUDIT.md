# EXAMGUARD AI — ML MODEL & VISION PIPELINE AUDIT

**Audit Date**: 2026-09-24  
**Auditor**: ML & Computer Vision Engineering Team  
**Scope**: All machine learning models, computer vision detectors, dependencies, datasets, backend inference pipelines, and API integrations in ExamGuard AI (PRJ 137).

---

## 1. Executive Summary

This audit assesses the state of machine learning, computer vision, data pipelines, and architectural modularity in ExamGuard AI. The existing project has established core prototypes (PyTorch MTCNN, YOLOv8n, MediaPipe FaceMesh heuristics, and synthetic-trained ensemble classifiers), but exhibits key architectural bottlenecks:

1. **Redundant Inference Passes**: Multiple detectors independently execute inference on identical video frames (e.g., MTCNN instantiated separately in both `face_detection.py` and `multi_face.py`).
2. **Missing Identity Verification**: Face detection exists, but candidate biometric identity verification (enrolled face representation vs. exam-time candidate verification) is not yet implemented.
3. **Coupled Business Logic & Detection**: Temporal event processing and alert debouncing are intertwined within detection classes rather than handled by a dedicated, decoupled temporal event engine.
4. **Dataset & Training Pipeline Gaps**: While pre-trained YOLOv8n weights exist for general COCO classes (`person`, `cell phone`, `book`, `laptop`), there is no dedicated, reproducible `ml/object_detection/` pipeline with data validation, corruption checking, split generation, metrics export (mAP@0.5, mAP@0.5:0.95), or model versioning.
5. **Score Interpretation**: Historical code mixed rule-based scoring with probability terminology; modern proctoring standards require configurable, evidence-backed *Risk Indicators* and *Suspicious Event Scores* rather than claiming definitive "cheating probability."

---

## 2. Comprehensive Model & Component Audit Matrix

| Existing Model / Component | Purpose | Framework & Architecture | Input Spec | Output Spec | Dataset Status | Training Status | Training Required? | Can Be Reused? | Problems Found | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face Detector** (`FaceDetector`) | Detects candidate face presence | PyTorch `facenet-pytorch` MTCNN | BGR/RGB Frame ($1280 \times 720$) | Bounding boxes, confidence probabilities | None (uses pre-trained cascade) | Pre-trained | No | Yes | Standalone MTCNN instance causes redundant inference; disappearance threshold (5s) hard-coded inside detector. | Unify into single-pass face pipeline; forward detection metadata to temporal behavior engine. |
| **Multi-Face Detector** (`MultiFaceDetector`) | Identifies unauthorized secondary persons | PyTorch `facenet-pytorch` MTCNN | BGR/RGB Frame | Secondary face bounding boxes, count | None (uses pre-trained cascade) | Pre-trained | No | Yes | Duplicates face detector MTCNN initialization and inference call. | Merge with primary face detector output. Evaluate `len(boxes) > 1` from single forward pass. |
| **Identity Verification** | Verifies candidate identity matches enrolled profile | Missing (Planned: `InceptionResnetV1` FaceNet) | Aligned face crop ($160 \times 160$) | 512-D normalized embedding, cosine similarity score | Needs enrollment database schema | Missing | No (use pre-trained VGGFace2 representation) | Yes (InceptionResnetV1) | Candidate enrollment does not capture or verify biometric embeddings. | Implement `ml/face/identity.py` using pre-trained `InceptionResnetV1`, configurable cosine threshold (default 0.65). |
| **Gaze & Eye Tracker** (`EyeTracker`) | Tracks eye direction & blink rate | MediaPipe `FaceMesh` + EAR calculation | RGB Frame | Eye Aspect Ratio (EAR), horizontal offset | None (Heuristic geometric) | Deterministic CV | No (Deterministic CV) | Yes | Gaze threshold ($\pm 15$ px) is absolute pixels rather than normalized to inter-pupillary/face width; ignores vertical gaze (looking down). | Implement normalized landmark geometry with temporal smoothing and vertical axis tracking. |
| **Head Pose Tracker** | Estimates 3D head orientation (Yaw, Pitch, Roll) | MediaPipe landmarks + OpenCV `solvePnP` | 3D Generic Facial Landmark Model | Euler angles ($^\circ$), pose category | None (Deterministic CV) | Deterministic CV | No (Deterministic CV) | Yes | Lacks temporal smoothing; single-frame jitters could trigger alerts; hard-coded in synthetic frame generator. | Implement mathematical `solvePnP` with Levenberg-Marquardt optimization and moving average filter. |
| **Mouth Activity Monitor** (`MouthMonitor`) | Detects talking, whispering, or vocal movement | MediaPipe `FaceMesh` (Lip indices 13, 14, 78, 306) | RGB Frame | Lip aperture ratio, width ratio | None (Geometric heuristic) | Deterministic CV | No (Deterministic CV) | Yes | Fixed thresholds ($0.03$) sensitive to candidate facial morphology; no baseline calibration. | Normalize lip aperture against nose-chin distance and apply temporal windowing. |
| **Object Detector** (`ObjectDetector`) | Detects cell phones, books, prohibited devices | Ultralytics YOLOv8 (`models/yolov8n.pt`) | Resized RGB Frame ($320 \times 320$) | Bounding boxes, class IDs, confidences | COCO pre-trained; missing proctoring domain split | Pre-trained (`yolov8n.pt`) | Fine-tuning recommended for exam objects (earphones, tablets, notes) | Yes | Pre-trained COCO weights lack specialized proctoring classes (e.g., earphones); no reproducible training/validation scripts in project. | Build modular `ml/object_detection/` pipeline with data validation, training, validation, export, and evaluation. |
| **Audio Voice Activity** (`AudioMonitor`) | Detects background acoustic talking | PyAudio Energy + Zero-Crossing Rate (ZCR) | 16kHz PCM audio chunks (512 samples) | Voice Activity Boolean, Energy level | None (Signal processing) | Deterministic DSP | No | Yes | PyAudio requires system C libraries; fails gracefully when hardware is absent. | Retain as secondary telemetry signal with configurable sensitivity. |
| **Malpractice Risk Model** (`malpractice_risk_model.pkl`) | Classifies candidate risk level | Scikit-Learn `RandomForestClassifier` (150 trees) | 12-D telemetry vector | Risk level (LOW, MODERATE, HIGH, CRITICAL), probability | Synthetic telemetry distributions | Trained (95.58% accuracy) | Fine-tuning on empirical exam logs | Yes | Trained on synthetic distributions; must be presented as a configurable risk indicator, not deterministic proof of cheating. | Reorganize into `ml/behavior/risk_engine.py` with transparent rule-weights and ML hybrid blending. |
| **Behavioral Anomaly Detector** (`candidate_anomaly_detector.pkl`) | Unsupervised outlier detection | Scikit-Learn `IsolationForest` + `RobustScaler` | 6-D behavioral time-series vector | Inlier/Outlier flag (-1, 1), anomaly score | Synthetic behavioral cadence | Trained (99.67% accuracy) | Retrainable with candidate logs | Yes | Requires real session buffering to compute accurate cadence windows. | Integrate into temporal event engine buffer. |

---

## 3. Detailed Component Breakdown

### 3.1 Hardware & Environment Dependencies
- **OS**: Windows (x86_64)
- **Python**: 3.14 (Anaconda base environment)
- **Core Frameworks**:
  - `torch`: `2.13.0+cpu` (CPU execution; CUDA fallback ready)
  - `torchvision`: `0.28.0+cpu`
  - `ultralytics`: `8.4.159`
  - `facenet-pytorch`: Installed (`MTCNN`, `InceptionResnetV1`)
  - `mediapipe`: `0.10.35`
  - `opencv-python`: `5.0.0`
  - `scikit-learn`: `1.9.0`
  - `scipy`: `1.18.0`
  - `pandas`: `3.0.3`
  - `numpy`: `2.4.6`
  - `flask`: `3.1.3`
  - `pillow`: `12.3.0`

### 3.2 Pre-Trained Weights Present
- `models/yolov8n.pt`: 6.2 MB official Ultralytics YOLOv8 nano checkpoint. Pre-trained on 80 COCO classes including `person` (0), `laptop` (63), `cell phone` (67), and `book` (73).
- `facenet-pytorch` internal checkpoints: MTCNN P-Net, R-Net, O-Net cached in PyTorch torch-hub; InceptionResnetV1 weights available on demand.

### 3.3 Components Requiring Machine Learning Training vs. Deterministic CV

| Component | Nature | Method Justification |
| :--- | :--- | :--- |
| **Object Detection** | Machine Learning (YOLOv8) | General visual variance in devices, lighting, and angles requires deep convolutional/transformer feature extractors. Pre-trained weights should be reused and fine-tuned. |
| **Face Representation** | Machine Learning (FaceNet) | Metric learning for face embeddings requires massive identity diversity (e.g. VGGFace2). Reusing pre-trained `InceptionResnetV1` is standard best practice; training from scratch is counter-productive. |
| **Head Pose** | Deterministic CV (Perspective-n-Point) | Geometric 3D-to-2D correspondence using canonical facial landmark coordinates via OpenCV `solvePnP` is mathematically exact, faster ($\ge 60$ FPS), and does not suffer from generalization bias. |
| **Eye Gaze** | Hybrid (Normalized Geometric CV + Regression) | Geometric Pupil Center Fraction (PCF) combined with normalized Eye Aspect Ratio (EAR) provides transparent, calibrated gaze boundaries without black-box bias. |
| **Event Engine** | Deterministic Temporal State Machine | Debouncing, sustained anomaly confirmation, and cooldown tracking must follow auditable, deterministic temporal logic rather than opaque neural predictions. |
| **Risk Scoring** | Configurable Hybrid Engine | Configurable event weights combined with ensemble anomaly scoring ensures explainable administrative auditing. |

---

## 4. Key Gaps Identified & Action Plan

1. **Gap**: No modular `ml/` package structure. Model logic is currently mixed between `src/detection/` and `src/dashboard/app.py`.
   - **Action**: Create clean, modular `ml/` hierarchy:
     - `ml/object_detection/` (train, validate, predict, inference)
     - `ml/face/` (detection, alignment, embeddings, identity verification)
     - `ml/head_pose/` (solvePnP estimator, temporal smoother)
     - `ml/gaze/` (normalized gaze estimator)
     - `ml/behavior/` (event engine, temporal state, risk engine)
     - `ml/inference/` (unified pipeline)

2. **Gap**: Missing candidate biometric identity verification during login/exam onboarding.
   - **Action**: Build `ml/face/identity.py` supporting reference face enrollment, feature embedding, and runtime cosine verification.

3. **Gap**: Missing dataset management and validation tools for object detection.
   - **Action**: Create standard `dataset/` directory with image corruption checking, label integrity validation, and `dataset.yaml`.

4. **Gap**: Missing evaluation metrics export (real precision, recall, mAP, FPS, latency).
   - **Action**: Create `evaluation/object_detection/` storing real `metrics.json`, `metrics.csv`, and performance plots.

5. **Gap**: Proctoring API endpoints are partially monolithic in Flask.
   - **Action**: Add explicit, standardized endpoints (`/api/proctoring/frame`, `/api/proctoring/start`, `/api/proctoring/stop`, `/api/proctoring/events`, `/api/proctoring/status`) while fully preserving existing web routes.

---

## 5. Audit Sign-Off
- **Architecture Stability**: High. Existing dashboard, CBT portal, and admin center templates are intact.
- **Safety**: All ML components will be modularized and verified with reproducible unit tests.
