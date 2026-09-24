"""
ExamGuard AI (PRJ 137) - Normalized Eye-Gaze Estimation Engine
Computes Eye Aspect Ratio (EAR) and inter-pupillary normalized horizontal/vertical gaze vectors.
Employs temporal moving-average smoothing to accommodate natural candidate saccades and blinks.
"""

import numpy as np
from collections import deque

class GazeEstimator:
    def __init__(self, ear_blink_thresh=0.20, horiz_ratio_thresh=0.22, vert_ratio_thresh=0.20, smoothing_window=5):
        self.ear_blink_thresh = ear_blink_thresh
        self.horiz_ratio_thresh = horiz_ratio_thresh
        self.vert_ratio_thresh = vert_ratio_thresh
        
        self.horiz_history = deque(maxlen=smoothing_window)
        self.vert_history = deque(maxlen=smoothing_window)
        self.ear_history = deque(maxlen=smoothing_window)
        
    def _calculate_ear(self, eye_pts):
        """Calculates Eye Aspect Ratio (EAR) from 6 landmark points."""
        pts = np.array(eye_pts)
        if len(pts) < 6:
            return 0.3
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3]) + 1e-6
        return float((v1 + v2) / (2.0 * h))
        
    def estimate_gaze(self, left_eye_pts, right_eye_pts, nose_pt=None):
        """
        Estimates gaze direction: CENTER, LEFT, RIGHT, UP, DOWN, UNKNOWN.
        """
        if left_eye_pts is None or right_eye_pts is None:
            return {
                'gaze': 'UNKNOWN',
                'ear': 0.3,
                'is_blinking': False,
                'is_looking_away': False
            }
            
        left_ear = self._calculate_ear(left_eye_pts)
        right_ear = self._calculate_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(avg_ear)
        smooth_ear = float(np.mean(self.ear_history))
        
        is_blinking = smooth_ear < self.ear_blink_thresh
        
        # Calculate eye centers
        left_center = np.mean(left_eye_pts, axis=0)
        right_center = np.mean(right_eye_pts, axis=0)
        inter_ocular_dist = np.linalg.norm(right_center - left_center) + 1e-6
        
        if nose_pt is not None:
            # Measure eye center relative to nose bridge normalized by inter-ocular distance
            mid_eyes = (left_center + right_center) / 2.0
            dx = (mid_eyes[0] - nose_pt[0]) / inter_ocular_dist
            # Anatomical baseline: eyes are naturally ~0.50 inter-ocular distances above nose tip
            raw_dy = (mid_eyes[1] - nose_pt[1]) / inter_ocular_dist
            dy = raw_dy - (-0.50)
        else:
            dx, dy = 0.0, 0.0
            
        self.horiz_history.append(dx)
        self.vert_history.append(dy)
        
        smooth_dx = float(np.mean(self.horiz_history))
        smooth_dy = float(np.mean(self.vert_history))
        
        # Classify gaze
        gaze = 'CENTER'
        is_looking_away = False
        
        if is_blinking:
            gaze = 'CENTER' # Natural blink is not an intentional look-away
        elif smooth_dy < -self.vert_ratio_thresh:
            gaze = 'UP'
            is_looking_away = True
        elif smooth_dy > self.vert_ratio_thresh:
            gaze = 'DOWN'
            is_looking_away = True
        elif smooth_dx < -self.horiz_ratio_thresh:
            gaze = 'LEFT'
            is_looking_away = True
        elif smooth_dx > self.horiz_ratio_thresh:
            gaze = 'RIGHT'
            is_looking_away = True
            
        return {
            'gaze': gaze,
            'ear': round(smooth_ear, 3),
            'horiz_offset': round(smooth_dx, 3),
            'vert_offset': round(smooth_dy, 3),
            'is_blinking': is_blinking,
            'is_looking_away': is_looking_away
        }
