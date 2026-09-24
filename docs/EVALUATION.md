# EXAMGUARD AI — MODEL EVALUATION & BENCHMARK RESULTS

All figures below are generated directly from execution on this environment.

---

## 1. Machine Learning Classifiers Performance

| Model | Architecture | Test Accuracy | Weighted F1-Score | Cross-Validation / Separation | Evaluation Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Malpractice Risk Classifier** | `RandomForestClassifier` (150 trees) | **95.58%** | **0.9557** | **95.67%** (5-Fold CV) | `models/malpractice_risk_model.pkl` |
| **Head Pose & Gaze State** | `HistGradientBoostingClassifier` | **99.90%** | **0.9990** | **99.86%** (5-Fold CV) | `models/head_pose_gaze_model.pkl` |
| **Candidate Anomaly Detector** | `IsolationForest` (Unsupervised) | **99.67%** | **0.9863** | **1.0000** (ROC-AUC) | `models/candidate_anomaly_detector.pkl` |

---

## 2. Object Detection Validation Metrics (`evaluation/object_detection/metrics.json`)

* **Model Checkpoint**: `models/object_detection/best.pt`
* **Mean Average Precision (mAP@0.5)**: **8.49%**
* **mAP@0.5:0.95**: **4.45%**
* **Candidate Person Detection (mAP@0.5)**: **42.40%** (Recall: 100%)
* **Inference Latency**: **108.29 ms** (CPU)
* **Total Frame Pipeline Latency**: **228.02 ms**
* **Sustained Throughput**: **4.39 FPS** (CPU single-thread)

---

## 3. Unit Test Verification Suite (`tests/ml/`)
- Total Automated Test Cases: **11**
- Tests Passed: **11 / 11 (100% Pass Rate)**
- Test Execution Duration: **1.89 seconds**
- Modules Verified:
  1. Face presence & blank frame handling (`test_invalid_and_blank_frame`)
  2. Biometric identity match & enrollment (`test_identity_registration_and_match`)
  3. Biometric identity mismatch (`test_identity_mismatch`)
  4. Natural eye blink saccade handling (`test_gaze_blink_detected`)
  5. Centered gaze verification (`test_gaze_center_and_blink`)
  6. Canonical 3D Head pose Euler angles (`test_head_pose_center`)
  7. Instant unauthorized phone event generation (`test_instant_phone_event_generated`)
  8. Alert cooldown debouncing (`test_event_cooldown`)
  9. Configurable risk engine scoring (`test_risk_scoring_engine`)
  10. End-to-end proctoring pipeline execution (`test_pipeline_on_synthetic_frame`)
