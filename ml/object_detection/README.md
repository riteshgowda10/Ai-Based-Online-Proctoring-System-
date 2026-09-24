# ExamGuard AI — Object Detection Module

## 1. Overview
The object detection module provides high-speed, multi-class visual object detection specifically tuned for online examination environments. Built upon Ultralytics YOLOv8, it detects candidates, electronic devices, unauthorized individuals, and prohibited study materials.

## 2. Supported Target Classes
- `0: person` — Exam candidate and secondary individuals in webcam view.
- `1: cell_phone` — Handheld mobile phones and screens.
- `2: laptop` — Authorized test computer or secondary computer monitor.
- `3: book` — Physical books, cheat sheets, notebooks.
- `4: earphone` — In-ear monitors, headphones, Bluetooth wireless buds.

## 3. Directory Layout
```
ml/object_detection/
├── dataset.yaml             # YOLO dataset configuration
├── validate_dataset.py      # Quality assurance & corruption checks
├── train.py                 # Reproducible training & fine-tuning
├── validate.py              # Benchmark metrics (mAP, F1, Latency, FPS)
├── predict.py               # Visual test prediction renderer
├── inference.py             # Modular real-time inference detector
└── README.md                # Documentation
```

## 4. Execution Commands
```powershell
# Validate dataset integrity
python -m ml.object_detection.validate_dataset

# Execute model training / fine-tuning
python -m ml.object_detection.train

# Benchmark on validation set (exports metrics.json and metrics.csv)
python -m ml.object_detection.validate

# Render test predictions with bounding boxes
python -m ml.object_detection.predict
```
