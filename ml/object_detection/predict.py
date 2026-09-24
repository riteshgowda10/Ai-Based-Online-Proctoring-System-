"""
ExamGuard AI (PRJ 137) - Object Detection Prediction & Visualization
Runs inference on images or test datasets, renders bounding boxes, class labels, and confidence tags.
"""

import os
import cv2
import glob
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
PRED_DIR = os.path.join(BASE_DIR, 'evaluation', 'object_detection', 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

def run_predictions(weights_path=None, source_dir=None, conf=0.45):
    if weights_path is None:
        best_pt = os.path.join(BASE_DIR, 'models', 'object_detection', 'best.pt')
        weights_path = best_pt if os.path.exists(best_pt) else os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
        
    if source_dir is None:
        source_dir = os.path.join(BASE_DIR, 'dataset', 'images', 'test')
        
    print(f"Running predictions with model: {weights_path}")
    print(f"Input image source: {source_dir}")
    
    model = YOLO(weights_path)
    images = glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png"))
    
    saved_paths = []
    for img_path in images:
        bname = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        
        out_path = os.path.join(PRED_DIR, f"pred_{bname}")
        cv2.imwrite(out_path, annotated)
        saved_paths.append(out_path)
        
    print(f"[OK] Generated {len(saved_paths)} prediction visualizations in: {PRED_DIR}")
    return saved_paths

if __name__ == '__main__':
    run_predictions()
