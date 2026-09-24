"""
ExamGuard AI (PRJ 137) - Temporal Behavior & Event Engine
Evaluates multi-frame streaming telemetry over temporal sliding windows.
Debounces transient blips, verifies sustained conditions, applies cooldowns, and generates structured events.
"""

import time
from datetime import datetime

class TemporalEventEngine:
    def __init__(self, config=None):
        cfg = config or {}
        temporal = cfg.get('temporal_debounce', {})
        
        self.face_missing_thresh = temporal.get('FACE_MISSING_MIN_SECONDS', 4.0)
        self.multiple_faces_thresh_frames = temporal.get('MULTIPLE_FACES_MIN_FRAMES', 5)
        self.looking_away_thresh = temporal.get('LOOKING_AWAY_MIN_SECONDS', 2.5)
        self.head_pose_thresh = temporal.get('HEAD_POSE_MIN_SECONDS', 2.0)
        self.cooldown_period = temporal.get('ALERT_COOLDOWN_SECONDS', 8.0)
        
        # State tracking
        self.face_missing_start = None
        self.multiple_faces_consecutive = 0
        self.looking_away_start = None
        self.head_pose_start = None
        
        # Cooldown timestamps {event_type: last_emitted_time}
        self.last_emitted_times = {}
        
        # History log
        self.event_history = []
        
    def _is_cooling_down(self, event_type, current_time):
        last_time = self.last_emitted_times.get(event_type)
        if last_time is None:
            return False
        return (current_time - last_time) < self.cooldown_period
        
    def process_frame_state(self, frame_id, detections, face_info, head_pose_info, gaze_info, identity_info):
        """
        Receives instant detections from all parallel modules and emits debounced events.
        """
        current_time = time.time()
        emitted_events = []
        
        # 1. FACE_MISSING Check (Sustained >= 4s)
        if not face_info.get('face_present', False):
            if self.face_missing_start is None:
                self.face_missing_start = current_time
            duration = current_time - self.face_missing_start
            if duration >= self.face_missing_thresh:
                if not self._is_cooling_down('FACE_MISSING', current_time):
                    event = {
                        'event_type': 'FACE_MISSING',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.95,
                        'duration': round(duration, 2),
                        'frame_id': str(frame_id),
                        'source_model': 'UnifiedFaceDetector',
                        'metadata': {'message': f'Candidate face missing for {duration:.1f}s'}
                    }
                    emitted_events.append(event)
                    self.last_emitted_times['FACE_MISSING'] = current_time
        else:
            self.face_missing_start = None
            
        # 2. MULTIPLE_FACES Check (Consecutive >= 5 frames)
        if face_info.get('face_count', 0) > 1:
            self.multiple_faces_consecutive += 1
            if self.multiple_faces_consecutive >= self.multiple_faces_thresh_frames:
                if not self._is_cooling_down('MULTIPLE_FACES', current_time):
                    event = {
                        'event_type': 'MULTIPLE_FACES',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': face_info.get('primary_confidence', 0.90),
                        'duration': round(self.multiple_faces_consecutive / 15.0, 2),
                        'frame_id': str(frame_id),
                        'source_model': 'UnifiedFaceDetector',
                        'metadata': {'face_count': face_info.get('face_count')}
                    }
                    emitted_events.append(event)
                    self.last_emitted_times['MULTIPLE_FACES'] = current_time
        else:
            self.multiple_faces_consecutive = 0
            
        # 3. IDENTITY_MISMATCH Check
        if identity_info and identity_info.get('status') == 'MISMATCH':
            if not self._is_cooling_down('IDENTITY_MISMATCH', current_time):
                event = {
                    'event_type': 'IDENTITY_MISMATCH',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 1.0 - identity_info.get('similarity', 0.0),
                    'duration': 0.0,
                    'frame_id': str(frame_id),
                    'source_model': 'IdentityVerifier',
                    'metadata': {'similarity': identity_info.get('similarity')}
                }
                emitted_events.append(event)
                self.last_emitted_times['IDENTITY_MISMATCH'] = current_time
                
        # 4. OBJECT DETECTIONS (Phones, Prohibited Items)
        for obj in detections:
            c_name = obj.get('normalized_name', '')
            if 'phone' in c_name:
                if not self._is_cooling_down('PHONE_DETECTED', current_time):
                    event = {
                        'event_type': 'PHONE_DETECTED',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': obj.get('confidence', 0.8),
                        'duration': 0.0,
                        'frame_id': str(frame_id),
                        'source_model': 'ExamObjectDetector',
                        'metadata': {'bbox': obj.get('bbox')}
                    }
                    emitted_events.append(event)
                    self.last_emitted_times['PHONE_DETECTED'] = current_time
            elif obj.get('is_prohibited', False):
                ev_name = 'PROHIBITED_OBJECT_DETECTED'
                if not self._is_cooling_down(ev_name, current_time):
                    event = {
                        'event_type': ev_name,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': obj.get('confidence', 0.8),
                        'duration': 0.0,
                        'frame_id': str(frame_id),
                        'source_model': 'ExamObjectDetector',
                        'metadata': {'object_class': obj.get('class'), 'bbox': obj.get('bbox')}
                    }
                    emitted_events.append(event)
                    self.last_emitted_times[ev_name] = current_time
                    
        # 5. HEAD_POSE_ANOMALY (Looking down or turned away sustained)
        if head_pose_info.get('is_deviated', False):
            if self.head_pose_start is None:
                self.head_pose_start = current_time
            duration = current_time - self.head_pose_start
            if duration >= self.head_pose_thresh:
                if not self._is_cooling_down('HEAD_POSE_ANOMALY', current_time):
                    event = {
                        'event_type': 'HEAD_POSE_ANOMALY',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.85,
                        'duration': round(duration, 2),
                        'frame_id': str(frame_id),
                        'source_model': 'HeadPoseEstimator',
                        'metadata': {
                            'pose': head_pose_info.get('pose'),
                            'yaw': head_pose_info.get('yaw'),
                            'pitch': head_pose_info.get('pitch')
                        }
                    }
                    emitted_events.append(event)
                    self.last_emitted_times['HEAD_POSE_ANOMALY'] = current_time
        else:
            self.head_pose_start = None
            
        # 6. LOOKING_AWAY (Gaze sustained away)
        if gaze_info.get('is_looking_away', False):
            if self.looking_away_start is None:
                self.looking_away_start = current_time
            duration = current_time - self.looking_away_start
            if duration >= self.looking_away_thresh:
                if not self._is_cooling_down('LOOKING_AWAY', current_time):
                    event = {
                        'event_type': 'LOOKING_AWAY',
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.82,
                        'duration': round(duration, 2),
                        'frame_id': str(frame_id),
                        'source_model': 'GazeEstimator',
                        'metadata': {'gaze': gaze_info.get('gaze')}
                    }
                    emitted_events.append(event)
                    self.last_emitted_times['LOOKING_AWAY'] = current_time
        else:
            self.looking_away_start = None
            
        for ev in emitted_events:
            self.event_history.append(ev)
            
        return emitted_events
        
    def get_all_events(self):
        return self.event_history
