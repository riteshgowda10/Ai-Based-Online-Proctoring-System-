# EXAMGUARD AI — OFFICIAL MODEL REGISTRY

**Last Updated**: 2026-09-24  
**Project**: PRJ 137 ExamGuard AI Proctoring System  

---

### Model 1: Proctoring Object Detector
* **Model Name**: `ExamGuard-YOLOv8n-Proctoring`
* **Version**: `2.0.0`
* **Framework**: Ultralytics YOLOv8 / PyTorch 2.13
* **File Location**: `models/object_detection/best.pt` (and `models/yolov8n.pt`)
* **Input Dimensions**: $640 \times 640 \times 3$ (RGB)
* **Classes**: `0: person`, `1: cell_phone`, `2: laptop`, `3: book`, `4: earphone`
* **Inference Latency**: ~108.3 ms (CPU) / ~8.5 ms (CUDA GPU)
* **Throughput**: 4.39 FPS (CPU) / >60 FPS (GPU)
* **License**: AGPL-3.0 / Commercial Compatible

---

### Model 2: Unified Face Detector
* **Model Name**: `MTCNN-Cascade-Detector`
* **Version**: `1.0.0`
* **Framework**: `facenet-pytorch` (PyTorch)
* **File Location**: Cached internally in PyTorch TorchHub
* **Input Dimensions**: Variable ($640 \times 480$ or $1280 \times 720$ BGR)
* **Output**: Primary face bounding box, confidence, secondary face count, 5 landmarks
* **Inference Latency**: ~28.0 ms (CPU)
* **License**: MIT

---

### Model 3: Biometric Identity Verifier
* **Model Name**: `FaceNet-InceptionResnetV1-VGGFace2`
* **Version**: `1.0.0`
* **Framework**: `facenet-pytorch` (PyTorch)
* **File Location**: Cached in PyTorch TorchHub
* **Input Dimensions**: $160 \times 160 \times 3$ Aligned Face Crop
* **Output**: 512-D L2-normalized biometric embedding vector
* **Similarity Metric**: Cosine Similarity ($\ge 0.65$ threshold)
* **Inference Latency**: ~32.4 ms (CPU)
* **License**: MIT

---

### Model 4: Multi-Modal Malpractice Risk Classifier
* **Model Name**: `ExamGuard-Malpractice-RF`
* **Version**: `1.2.0`
* **Framework**: Scikit-Learn 1.9.0 (`RandomForestClassifier`, 150 trees)
* **File Location**: `models/malpractice_risk_model.pkl`
* **Input Dimensions**: 12-D multi-modal telemetry vector
* **Validation Accuracy**: 95.58% (5-Fold CV: 95.67%)
* **Classes**: `LOW`, `MODERATE`, `HIGH`, `CRITICAL`
* **Inference Latency**: < 0.5 ms
* **License**: Proprietary PRJ 137

---

### Model 5: Head Pose & Gaze Deviation Classifier
* **Model Name**: `ExamGuard-PoseGaze-GB`
* **Version**: `1.1.0`
* **Framework**: Scikit-Learn (`HistGradientBoostingClassifier`)
* **File Location**: `models/head_pose_gaze_model.pkl`
* **Input Dimensions**: 8-D landmark geometric features
* **Validation Accuracy**: 99.90% (5-Fold CV: 99.86%)
* **Classes**: `Center`, `Turned Left`, `Turned Right`, `Looking Down (Phone Risk)`
* **Inference Latency**: < 0.4 ms
* **License**: Proprietary PRJ 137

---

### Model 6: Candidate Behavioral Anomaly Detector
* **Model Name**: `ExamGuard-Anomaly-IForest`
* **Version**: `1.0.0`
* **Framework**: Scikit-Learn (`IsolationForest`, 150 estimators, RobustScaler)
* **File Location**: `models/candidate_anomaly_detector.pkl`
* **Input Dimensions**: 6-D behavioral cadence vector (typing, tab switches, gaze jitter, audio energy)
* **ROC-AUC**: 1.0000 | Inlier/Outlier separation
* **Inference Latency**: < 0.6 ms
* **License**: Proprietary PRJ 137
