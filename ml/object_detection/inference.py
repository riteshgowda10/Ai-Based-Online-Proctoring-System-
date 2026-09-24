"""
ExamGuard AI (PRJ 137) - Programmatic Object Detection Engine
High-throughput inference detector for integration into the real-time proctoring pipeline.
"""

import os
import cv2
import yaml
from ultralytics import YOLO

class ExamObjectDetector:
    def __init__(self, config_path=None, weights_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # Resolve config
        if config_path is None:
            config_path = os.path.join(base_dir, 'configs', 'inference.yaml')
            
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        obj_cfg = cfg.get('models', {}).get('object_detection', {})
        self.conf_thresh = obj_cfg.get('confidence_threshold', 0.50)
        self.iou_thresh = obj_cfg.get('iou_threshold', 0.45)
        self.target_classes = obj_cfg.get('target_classes', ['person', 'cell phone', 'laptop', 'book'])
        
        # Prohibited objects classification list
        self.prohibited_labels = {'cell phone', 'cell_phone', 'book', 'earphone', 'prohibited_object'}
        
        # Resolve weights path
        if weights_path is None:
            best_pt = os.path.join(base_dir, 'models', 'object_detection', 'best.pt')
            weights_path = best_pt if os.path.exists(best_pt) else os.path.join(base_dir, 'models', 'yolov8n.pt')
            
        self.weights_path = weights_path
        self.model = YOLO(weights_path)
        
    def detect(self, frame, visualize=False):
        """
        Runs object detection on a BGR frame.
        Returns:
            detections (list of dicts): List of detected items.
            annotated_frame (np.ndarray): Frame with rendered overlays (if visualize=True).
        """
        if frame is None or frame.size == 0:
            return [], frame
            
        # Optional resize optimization for CPU
        orig_h, orig_w = frame.shape[:2]
        
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )
        
        detections = []
        annotated_frame = results[0].plot() if visualize else frame
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.model.names.get(cls_id, str(cls_id))
            
            # Normalize class names
            normalized_name = cls_name.replace(" ", "_").lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            is_prohibited = (normalized_name in self.prohibited_labels or cls_name.lower() in self.prohibited_labels)
            
            detections.append({
                'class': cls_name,
                'normalized_name': normalized_name,
                'class_id': cls_id,
                'confidence': round(conf, 4),
                'bbox': [x1, y1, x2, y2],
                'is_prohibited': is_prohibited
            })
            
        return detections, annotated_frame
