"""
ExamGuard AI - Dataset Quality Assurance & Validator
Validates image files, detects corruptions, checks label syntax, and computes class distributions.
"""

import os
import cv2
import glob
from PIL import Image

def validate_split(dataset_dir, split_name, num_classes=5):
    img_dir = os.path.join(dataset_dir, 'images', split_name)
    lbl_dir = os.path.join(dataset_dir, 'labels', split_name)
    
    images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    labels = glob.glob(os.path.join(lbl_dir, '*.txt'))
    
    corrupted_images = []
    missing_labels = []
    invalid_labels = []
    class_counts = {i: 0 for i in range(num_classes)}
    total_annotations = 0
    
    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Test image integrity
        try:
            with Image.open(img_path) as img:
                img.verify()
            mat = cv2.imread(img_path)
            if mat is None or mat.size == 0:
                corrupted_images.append(img_path)
        except Exception:
            corrupted_images.append(img_path)
            continue
            
        # Check corresponding label file
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        if not os.path.exists(lbl_path):
            missing_labels.append(img_path)
            continue
            
        # Parse annotations
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    invalid_labels.append(f"{lbl_path}:L{line_idx} - expected 5 elements, got {len(parts)}")
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    if cls_id < 0 or cls_id >= num_classes:
                        invalid_labels.append(f"{lbl_path}:L{line_idx} - invalid class {cls_id}")
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        invalid_labels.append(f"{lbl_path}:L{line_idx} - out of bounds [0, 1] {coords}")
                        
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_annotations += 1
                except ValueError:
                    invalid_labels.append(f"{lbl_path}:L{line_idx} - parse error")
                    
    return {
        'split': split_name,
        'image_count': len(images),
        'label_count': len(labels),
        'corrupted_images': corrupted_images,
        'missing_labels': missing_labels,
        'invalid_labels': invalid_labels,
        'class_counts': class_counts,
        'total_annotations': total_annotations,
        'is_valid': len(corrupted_images) == 0 and len(missing_labels) == 0 and len(invalid_labels) == 0
    }

def run_dataset_validation(dataset_dir="dataset"):
    class_names = ['person', 'cell_phone', 'laptop', 'book', 'earphone']
    print("=" * 65)
    print("EXAMGUARD AI — OBJECT DETECTION DATASET VALIDATION REPORT")
    print("=" * 65)
    
    all_valid = True
    overall_counts = {i: 0 for i in range(len(class_names))}
    total_imgs = 0
    
    for split in ['train', 'val', 'test']:
        res = validate_split(dataset_dir, split, num_classes=len(class_names))
        print(f"\n[Split: {split.upper()}]")
        print(f"  Images verified : {res['image_count']} valid, {len(res['corrupted_images'])} corrupted")
        print(f"  Labels verified : {res['label_count']} files, {res['total_annotations']} bounding boxes")
        print(f"  Missing labels  : {len(res['missing_labels'])}")
        print(f"  Syntax errors   : {len(res['invalid_labels'])}")
        print("  Class distribution:")
        for cid, name in enumerate(class_names):
            cnt = res['class_counts'].get(cid, 0)
            overall_counts[cid] += cnt
            print(f"    - Class {cid} ({name:11s}): {cnt:3d} boxes")
            
        if not res['is_valid']:
            all_valid = False
        total_imgs += res['image_count']
        
    print("\n" + "-" * 65)
    print(f"OVERALL SUMMARY: {total_imgs} total images across all splits.")
    print("Aggregate Class Annotations:")
    for cid, name in enumerate(class_names):
        print(f"  - {name:12s}: {overall_counts[cid]:4d} instances")
    print(f"DATASET INTEGRITY STATUS: {'PASSED [OK]' if all_valid else 'FAILED [ERRORS DETECTED]'}")
    print("=" * 65)
    return all_valid

if __name__ == '__main__':
    run_dataset_validation()
