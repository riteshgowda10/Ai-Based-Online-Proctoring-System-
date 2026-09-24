"""
ExamGuard AI (PRJ 137) - Unified Real-Time Inference Pipeline
Orchestrates parallel vision modules, temporal behavior debouncing, and risk indicator scoring.
"""

import os
import cv2
import time
from datetime import datetime

from ml.object_detection.inference import ExamObjectDetector
from ml.face.detector import UnifiedFaceDetector
from ml.face.identity import IdentityVerifier
from ml.head_pose.estimator import HeadPoseEstimator
from ml.gaze.estimator import GazeEstimator
from ml.behavior.event_engine import TemporalEventEngine
from ml.behavior.risk_engine import RiskEngine

class RealtimeProctoringPipeline:
    def __init__(self, config_dir=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        self.object_detector = ExamObjectDetector()
        self.face_detector = UnifiedFaceDetector()
        self.identity_verifier = IdentityVerifier()
        self.head_pose_estimator = HeadPoseEstimator()
        self.gaze_estimator = GazeEstimator()
        self.event_engine = TemporalEventEngine()
        self.risk_engine = RiskEngine()
        
        self.frame_counter = 0
        
    def process_frame(self, frame, student_id="STUDENT_001", browser_telemetry=None):
        """
        Receives raw BGR camera frame and returns complete structured proctoring assessment.
        """
        self.frame_counter += 1
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            return {
                'timestamp': datetime.now().isoformat(),
                'frame_id': self.frame_counter,
                'status': 'INVALID_FRAME',
                'face_count': 0,
                'identity_match': False,
                'head_pose': {},
                'gaze': {},
                'objects': [],
                'events': [],
                'risk_indicator': 0,
                'risk_level': 'LOW',
                'latency_ms': 0.0
            }
            
        h, w = frame.shape[:2]
        
        # 1. Parallel Module: Object Detection
        objects, _ = self.object_detector.detect(frame)
        
        # 2. Parallel Module: Face Detection
        face_info = self.face_detector.detect(frame)
        
        # 3. Facial Landmarks & Head Pose & Gaze
        head_pose = {'pose': 'unknown', 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'is_deviated': False}
        gaze = {'gaze': 'UNKNOWN', 'ear': 0.3, 'is_blinking': False, 'is_looking_away': False}
        
        if face_info['face_present'] and len(face_info.get('landmarks', [])) >= 5:
            lms = face_info['landmarks']
            # MTCNN provides 5 landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
            # Construct 6-point image array for solvePnP
            # model_points: Nose(0), Chin(1), LeftEye(2), RightEye(3), LeftMouth(4), RightMouth(5)
            # We approximate chin below nose
            nose = lms[2]
            left_eye = lms[0]
            right_eye = lms[1]
            left_mouth = lms[3]
            right_mouth = lms[4]
            chin = [nose[0], nose[1] + (nose[1] - (left_eye[1] + right_eye[1])/2.0)]
            
            pnp_points = [nose, chin, left_eye, right_eye, left_mouth, right_mouth]
            head_pose = self.head_pose_estimator.estimate_pose((h, w), pnp_points)
            
            # Gaze approximation
            gaze = self.gaze_estimator.estimate_gaze([left_eye]*6, [right_eye]*6, nose_pt=nose)
            
        # 4. Identity Verification
        match, similarity, id_status = self.identity_verifier.verify_identity(student_id, frame)
        identity_info = {
            'match': match,
            'similarity': similarity,
            'status': id_status
        }
        
        # 5. Temporal Event Engine
        new_events = self.event_engine.process_frame_state(
            frame_id=self.frame_counter,
            detections=objects,
            face_info=face_info,
            head_pose_info=head_pose,
            gaze_info=gaze,
            identity_info=identity_info
        )
        
        # 6. Active Conditions for Risk Engine
        telemetry = browser_telemetry or {}
        active_conditions = {
            'FACE_MISSING': not face_info['face_present'],
            'MULTIPLE_FACES': face_info['face_count'] > 1,
            'IDENTITY_MISMATCH': not match,
            'PHONE_DETECTED': any('phone' in o.get('normalized_name', '') for o in objects),
            'PROHIBITED_OBJECT_DETECTED': any(o.get('is_prohibited', False) for o in objects),
            'HEAD_POSE_ANOMALY': head_pose.get('is_deviated', False),
            'LOOKING_AWAY': gaze.get('is_looking_away', False),
            'BROWSER_TAB_SWITCH': telemetry.get('tab_switches', 0),
            'CLIPBOARD_COPY_PASTE': telemetry.get('copy_attempts', 0),
            'SCREENSHOT_INTERCEPT': telemetry.get('screenshot_attempts', 0)
        }
        
        risk_result = self.risk_engine.calculate_risk(active_conditions)
        
        total_latency_ms = round((time.time() - start_time) * 1000.0, 2)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'frame_id': self.frame_counter,
            'status': 'OK',
            'face_count': face_info['face_count'],
            'primary_face_confidence': face_info['primary_confidence'],
            'identity_match': match,
            'identity_similarity': similarity,
            'head_pose': head_pose,
            'gaze': gaze,
            'objects': objects,
            'events': new_events,
            'risk_indicator': risk_result['proctoring_risk_indicator'],
            'integrity_index': risk_result['candidate_integrity_index'],
            'risk_level': risk_result['risk_level'],
            'risk_contributors': risk_result['contributors'],
            'latency_ms': total_latency_ms
        }
