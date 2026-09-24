# ExamGuard AI — Model Weights Directory

This directory stores trained and fine-tuned machine learning checkpoints, vision weights, and scikit-learn ensemble pipelines used by the proctoring engine.

## Checkpoint Registry

| Model File | Architecture / Framework | Size | Purpose | Source / Origin |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | Ultralytics YOLOv8 nano (PyTorch) | 6.2 MB | Base pre-trained visual detector for person, cell phone, laptop, book | Official Ultralytics Release |
| `object_detection/best.pt` | Fine-tuned YOLOv8n (PyTorch) | 6.2 MB | Proctoring domain fine-tuned checkpoint for exam objects and candidate presence | Trained via `ml/object_detection/train.py` |
| `malpractice_risk_model.pkl` | Scikit-Learn `RandomForestClassifier` (150 trees) | 4.3 MB | Telemetry risk level and candidate cheating risk scoring | Trained via `src/training/train_models.py` |
| `head_pose_gaze_model.pkl` | Scikit-Learn `HistGradientBoosting` | 1.2 MB | Landmark geometric posture & attention state classifier | Trained via `src/training/train_models.py` |
| `candidate_anomaly_detector.pkl` | Scikit-Learn `IsolationForest` | 1.3 MB | Unsupervised candidate behavioral anomaly & cadence outlier detection | Trained via `src/training/train_models.py` |

## Model Loading Instructions
All models are automatically discovered and loaded dynamically through:
- Programmatic Inference: `ml.inference.pipeline.RealtimeProctoringPipeline`
- Object Detection: `ml.object_detection.inference.ExamObjectDetector`
- Identity Verification: `ml.face.identity.IdentityVerifier`

If downloading weights manually on a new environment:
```powershell
# Re-train or export weights on any new machine:
python -m ml.object_detection.train
python src/training/train_models.py
```
