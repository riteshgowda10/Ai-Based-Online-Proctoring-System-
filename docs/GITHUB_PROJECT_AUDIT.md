# EXAMGUARD AI — GITHUB PRE-UPLOAD PROJECT AUDIT

**Audit Date**: 2026-09-24  
**Target Repository**: `https://github.com/riteshgowda10`  
**Author**: Ritesh A S  
**System**: Complete AI-Based Online Examination Proctoring System  

---

## 1. Directory & Component Mapping

| Subsystem | Exact Location | Files / Technologies Included |
| :--- | :--- | :--- |
| **Frontend** | `src/dashboard/templates/` | HTML5, Tailwind CSS, FontAwesome, JavaScript Web APIs, Admin Control Center (`admin.html`), Candidate Exam Interface (`exam.html`), Login/Registration (`login.html`), Instructions (`instructions.html`), Live Dashboard (`dashboard.html`) |
| **Backend** | `src/dashboard/app.py`, `src/utils/` | Flask 3.1.3 WSGI Server, REST APIs, Session Management, MJPEG Live Stream, Telemetry Logger, Alert System |
| **Machine Learning** | `ml/` & `src/detection/` | `ml/object_detection/`, `ml/face/`, `ml/head_pose/`, `ml/gaze/`, `ml/behavior/`, `ml/inference/` |
| **Database** | `database/` | `database/schema.sql`, `database/README.md` |
| **Configurations** | `configs/` & `config/` | `configs/training.yaml`, `configs/inference.yaml`, `configs/proctoring.yaml`, `config/config.yaml`, `.env.example` |
| **Tests** | `tests/ml/` | `test_face_identity.py`, `test_head_pose_gaze.py`, `test_objects_behavior.py`, `test_pipeline.py` (11 / 11 Passing) |
| **Documentation** | `docs/` & Root | `README.md`, `ML_MODEL_AUDIT.md`, `ML_ARCHITECTURE.md`, `DATASET_GUIDE.md`, `TRAINING_GUIDE.md`, `EVALUATION.md`, `INFERENCE_GUIDE.md`, `MODEL_LIMITATIONS.md`, `FINAL_ML_REPORT.md` |
| **Model Weights** | `models/` | `yolov8n.pt` (6.2 MB), `best.pt` (6.2 MB), `malpractice_risk_model.pkl` (4.3 MB), `head_pose_gaze_model.pkl` (1.2 MB), `candidate_anomaly_detector.pkl` (1.3 MB), `models/README.md`, `models/MODEL_REGISTRY.md` |
| **Datasets** | `dataset/` & `datasets/` | `dataset/images/`, `dataset/labels/` (6.8 MB total), `datasets/README.md` |

---

## 2. Commit Whitelist vs. Blacklist

### Files that SHOULD be committed:
- All source code in `src/`, `ml/`, `database/`, `tests/`
- All configuration files in `configs/`, `config/`, `.env.example`, `requirements.txt`
- All documentation in `docs/`, `models/`, `datasets/`, `README.md`, `LICENSE`
- Production model weights $< 10$ MB: `models/yolov8n.pt`, `models/object_detection/best.pt`, `models/*.pkl`
- Verification micro-dataset: `dataset/` (6.8 MB)

### Files that MUST NOT be committed (Enforced by `.gitignore`):
- Local secrets, passwords, `.env`
- Python bytecode and caches (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `.venv/`, `env/`)
- Node modules (`node_modules/`)
- Log files (`logs/*.log`, `alerts.log`)
- Temporary/session screen captures (`recordings/`, `captures/`, `generated_reports/*.pdf`)
- OS files (`Thumbs.db`, `.DS_Store`, `.vscode/`, `.idea/`)
