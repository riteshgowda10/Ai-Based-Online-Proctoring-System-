"""
ExamGuard AI (PRJ 137) - Unified Face Detector
Performs single-pass detection of primary and secondary faces using PyTorch MTCNN.
Avoids redundant inference passes.
"""

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

class UnifiedFaceDetector:
    def __init__(self, config=None, min_face_size=40, thresholds=(0.6, 0.7, 0.7)):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            post_process=False,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            device=self.device
        )
        self.min_confidence = 0.80
        
    def detect(self, frame):
        """
        Processes BGR frame and returns face detections.
        Returns:
            dict: {
                'face_present': bool,
                'face_count': int,
                'primary_box': [x1, y1, x2, y2] or None,
                'primary_confidence': float,
                'secondary_boxes': list of [x1, y1, x2, y2],
                'landmarks': list of landmark coordinates
            }
        """
        if frame is None or frame.size == 0:
            return {
                'face_present': False,
                'face_count': 0,
                'primary_box': None,
                'primary_confidence': 0.0,
                'secondary_boxes': [],
                'landmarks': []
            }
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.detector.detect(rgb, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return {
                'face_present': False,
                'face_count': 0,
                'primary_box': None,
                'primary_confidence': 0.0,
                'secondary_boxes': [],
                'landmarks': []
            }
            
        # Filter high confidence faces
        valid_faces = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is not None and prob >= self.min_confidence:
                valid_faces.append({
                    'box': [int(b) for b in box],
                    'prob': float(prob),
                    'landmarks': landmarks[i].tolist() if landmarks is not None and i < len(landmarks) else []
                })
                
        if len(valid_faces) == 0:
            return {
                'face_present': False,
                'face_count': 0,
                'primary_box': None,
                'primary_confidence': 0.0,
                'secondary_boxes': [],
                'landmarks': []
            }
            
        # Primary face is the largest bounding box area
        valid_faces.sort(key=lambda f: (f['box'][2]-f['box'][0]) * (f['box'][3]-f['box'][1]), reverse=True)
        primary = valid_faces[0]
        secondaries = [f['box'] for f in valid_faces[1:]]
        
        return {
            'face_present': True,
            'face_count': len(valid_faces),
            'primary_box': primary['box'],
            'primary_confidence': round(primary['prob'], 4),
            'secondary_boxes': secondaries,
            'landmarks': primary['landmarks']
        }
