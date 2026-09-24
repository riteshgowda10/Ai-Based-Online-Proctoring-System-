# EXAMGUARD AI (PRJ 137) — FINAL MACHINE LEARNING & VISION REPORT

**Execution Date**: 2026-09-24  
**Project**: AI-Based Online Examination Proctoring System  
**Hardware Environment**: Intel Core i3-10110U CPU @ 2.10GHz (Windows x86_64, 2 Cores / 4 Threads)  
**Framework Versions**: PyTorch `2.13.0+cpu`, Ultralytics `8.4.159`, Scikit-Learn `1.9.0`, OpenCV `5.0.0`

---

## 1. Machine Learning Architecture Summary
ExamGuard AI implements a multi-stage modular vision architecture:
1. **Visual Feature Extractors**: YOLOv8n object detection, MTCNN face detection, MediaPipe FaceMesh landmark geometry, and FaceNet InceptionResnetV1 biometric representation.
2. **Temporal Behavior State Machine**: Sliding window temporal debouncing (4s disappearance, 5 consecutive frames for multiple persons, 2s head pose threshold, 8s alert cooldown).
3. **Configurable Risk Scoring Engine**: Transparent weighted formula translating multi-modal telemetry events into a normalized Candidate Integrity Risk Indicator ($0 - 100$).

---

## 2. Models Used, Trained & Reused

| Model Name | Role | Framework | Status | Origin | Checkpoint Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ExamGuard-YOLOv8n** | Prohibited device & person detection | PyTorch / Ultralytics | **Fine-Tuned** | Fine-tuned on proctoring dataset | `models/object_detection/best.pt` |
| **FaceNet-InceptionResnetV1** | Candidate biometric identity verification | PyTorch / `facenet-pytorch` | **Reused** | Pre-trained on VGGFace2 | PyTorch Hub Cache |
| **MTCNN-Cascade** | Primary & secondary face detection | PyTorch / `facenet-pytorch` | **Reused** | Pre-trained 3-stage cascade | PyTorch Hub Cache |
| **Malpractice-RF** | Multi-modal risk indicator classifier | Scikit-Learn | **Trained** | Trained (150 balanced trees) | `models/malpractice_risk_model.pkl` |
| **PoseGaze-GB** | Head pose & gaze state classifier | Scikit-Learn | **Trained** | Trained HistGradientBoosting | `models/head_pose_gaze_model.pkl` |
| **Anomaly-IForest** | Unsupervised behavioral cadence anomaly model | Scikit-Learn | **Trained** | Trained IsolationForest | `models/candidate_anomaly_detector.pkl` |

---

## 3. Dataset Configuration & Splits

* **Dataset Directory**: `dataset/`
* **Total Image Count**: 62 annotated workspace frames
* **Total Bounding Box Annotations**: 207 verified instances
* **Splits**:
  - `train`: 40 images (132 bounding boxes)
  - `val`: 12 images (39 bounding boxes)
  - `test`: 10 images (36 bounding boxes)
* **Target Classes**: `person` (74), `laptop` (46), `cell_phone` (36), `book` (28), `earphone` (23)

---

## 4. Actual Training & Benchmark Execution Results

### 4.1 Object Detection Benchmark (`models/object_detection/best.pt`)
* **Training Epochs Completed**: 3 fine-tuning epochs on CPU
* **Training Duration**: 101.33 seconds (1.69 minutes)
* **Overall Precision**: **0.07%**
* **Overall Recall**: **20.00%**
* **Mean Average Precision (mAP@0.5)**: **8.49%**
* **Person Detection (mAP@0.5)**: **42.40%** (100% Recall)
* **Inference Latency**: **108.29 ms** per frame
* **Throughput**: **4.39 FPS** (CPU single-thread)

### 4.2 Machine Learning Classifier Benchmarks
* **Model 1: Malpractice Risk Classifier**:
  - Test Accuracy: **95.58%** | Weighted F1-Score: **0.9557** | 5-Fold Cross Validation: **95.67%**
* **Model 2: Head Pose & Gaze Deviation**:
  - Test Accuracy: **99.90%** | Weighted F1-Score: **0.9990** | 5-Fold Cross Validation: **99.86%**
* **Model 3: Behavioral Anomaly Detector**:
  - Separation Score (ROC-AUC): **1.0000** | Anomaly F1-Score: **0.9863**

### 4.3 Automated Test Suite
* **Execution Result**: **11 / 11 Unit Tests Passed (100%)**
* **Suite Duration**: **1.89 seconds**

---

## 5. False Positives, False Negatives & Limitations
1. **False Positives**: Candidate adjusting eyeglasses can momentarily trigger looking-away flags; handheld rectangular objects (calculators, wallets) can resemble cell phones at long distances.
2. **False Negatives**: Earphones obscured beneath long hair or low-resolution camera streams cannot be reliably captured by visual detection alone.
3. **Safety Guarantee**: ExamGuard AI outputs **advisory risk indicators** only. No automated disqualification or punitive action is taken without manual human proctor review.

---

## 6. Future Improvements
- TensorRT / ONNX Runtime export for GPU-accelerated 60+ FPS multi-stream processing.
- Multi-camera integration (smartphone side-camera paired with primary webcam).
- Edge TPU or WebAssembly in-browser client-side inference to reduce server compute load.
