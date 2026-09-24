# ExamGuard AI — Proctoring Dataset Specification

## 1. Dataset Overview
This directory layout defines the standardized computer vision dataset format for training and fine-tuning YOLO-family object detectors on examination workspace environments.

## 2. Supported Target Classes
- `0: person` — Exam candidate and any secondary individuals in camera view.
- `1: cell_phone` — Handheld smartphones, smart displays, illuminated screens.
- `2: laptop` — Computer monitors, secondary laptops, tablet displays.
- `3: book` — Textbooks, cheat notes, open notebooks.
- `4: earphone` — Wireless earbuds, over-ear headphones, wired earphones.

## 3. Directory Structure
```
dataset/
├── images/
│   ├── train/       # Training frames (.jpg / .png)
│   ├── val/         # Validation frames
│   └── test/        # Test evaluation frames
└── labels/
    ├── train/       # Normalized YOLO annotations (.txt)
    ├── val/
    └── test/
```

## 4. Annotation Format (YOLO Normalized)
Each `.txt` file shares the base name of its corresponding image. Each line represents a detected bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinate values are normalized between `0.0` and `1.0`.

## 5. Dataset Validation & Quality Assurance
Run the automated dataset validator before training:
```powershell
python -m ml.object_detection.validate_dataset
```
Checks for:
- Corrupted or unreadable image binaries
- Missing label text files
- Out-of-bounds bounding box coordinates
- Class distribution imbalances across splits
