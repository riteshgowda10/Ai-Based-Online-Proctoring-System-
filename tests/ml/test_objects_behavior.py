"""
Unit Tests: Object Detection, Temporal Event Engine & Risk Indicator
Tests: phone detection, prohibited objects, missing face debounce, cooldown, risk scoring
"""

import os
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.behavior.event_engine import TemporalEventEngine
from ml.behavior.risk_engine import RiskEngine

class TestObjectsAndBehavior(unittest.TestCase):
    def setUp(self):
        self.event_engine = TemporalEventEngine()
        self.risk_engine = RiskEngine()
        
    def test_instant_phone_event_generated(self):
        detections = [
            {'class': 'cell phone', 'normalized_name': 'cell_phone', 'confidence': 0.88, 'bbox': [100, 100, 200, 200], 'is_prohibited': True}
        ]
        face_info = {'face_present': True, 'face_count': 1, 'primary_confidence': 0.95}
        head_pose = {'pose': 'looking_center', 'is_deviated': False}
        gaze = {'gaze': 'CENTER', 'is_looking_away': False}
        identity = {'status': 'MATCH', 'similarity': 0.9}
        
        events = self.event_engine.process_frame_state(
            frame_id=1,
            detections=detections,
            face_info=face_info,
            head_pose_info=head_pose,
            gaze_info=gaze,
            identity_info=identity
        )
        
        self.assertTrue(any(e['event_type'] == 'PHONE_DETECTED' for e in events))
        
    def test_event_cooldown(self):
        detections = [
            {'class': 'cell phone', 'normalized_name': 'cell_phone', 'confidence': 0.88, 'bbox': [100, 100, 200, 200], 'is_prohibited': True}
        ]
        face_info = {'face_present': True, 'face_count': 1}
        head_pose = {'pose': 'looking_center', 'is_deviated': False}
        gaze = {'gaze': 'CENTER', 'is_looking_away': False}
        identity = {'status': 'MATCH'}
        
        # Frame 1: Should fire
        ev1 = self.event_engine.process_frame_state(1, detections, face_info, head_pose, gaze, identity)
        self.assertTrue(any(e['event_type'] == 'PHONE_DETECTED' for e in ev1))
        
        # Frame 2 immediately after: Should be blocked by cooldown
        ev2 = self.event_engine.process_frame_state(2, detections, face_info, head_pose, gaze, identity)
        self.assertFalse(any(e['event_type'] == 'PHONE_DETECTED' for e in ev2))
        
    def test_risk_scoring_engine(self):
        # Benign test
        benign_conditions = {'FACE_MISSING': False, 'MULTIPLE_FACES': False, 'PHONE_DETECTED': False}
        res_benign = self.risk_engine.calculate_risk(benign_conditions)
        self.assertEqual(res_benign['risk_level'], 'LOW')
        self.assertTrue(res_benign['candidate_integrity_index'] >= 90)
        
        # High-risk conditions
        critical_conditions = {
            'FACE_MISSING': True,
            'MULTIPLE_FACES': True,
            'PHONE_DETECTED': True,
            'BROWSER_TAB_SWITCH': 3
        }
        res_crit = self.risk_engine.calculate_risk(critical_conditions)
        self.assertEqual(res_crit['risk_level'], 'CRITICAL')
        self.assertTrue(res_crit['proctoring_risk_indicator'] >= 75)
        self.assertTrue(len(res_crit['contributors']) >= 3)

if __name__ == '__main__':
    unittest.main()
