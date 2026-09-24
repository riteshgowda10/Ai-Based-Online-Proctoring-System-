# EXAMGUARD AI — REPRODUCIBLE MODEL TRAINING GUIDE

## 1. Quick-Start Training Commands

### Train All 3 Machine Learning Proctoring Models
```powershell
# Trains Random Forest Risk Model, HistGradientBoosting Posture Model, Isolation Forest Anomaly Model
python src/training/train_models.py
```

### Validate Object Detection Dataset
```powershell
python -m ml.object_detection.validate_dataset
```

### Fine-Tune YOLOv8 Object Detector
```powershell
# Executes reproducible fine-tuning loop using configs/training.yaml
python -m ml.object_detection.train
```

### Run Benchmark Evaluation & Latency Profiling
```powershell
# Computes mAP@0.5, mAP@0.5:0.95, per-class metrics, FPS, latency
python -m ml.object_detection.validate
```

### Render Prediction Previews
```powershell
python -m ml.object_detection.predict
```

## 2. Training Hyperparameters (`configs/training.yaml`)
- **Random Seed**: `42` (Deterministic data splitting & weight initialization)
- **Batch Size**: `8`
- **Initial Learning Rate**: `0.005`
- **Optimizer**: `SGD` (momentum `0.937`, weight decay `0.0005`)
- **Image Size**: $640 \times 640$
