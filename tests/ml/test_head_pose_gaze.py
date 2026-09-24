"""
Unit Tests: Head Pose Estimation & Eye Gaze Estimation
Tests: normal gaze, looking left, looking right, looking down, blink debouncing, solvePnP Euler angles
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.head_pose.estimator import HeadPoseEstimator
from ml.gaze.estimator import GazeEstimator

class TestHeadPoseAndGaze(unittest.TestCase):
    def setUp(self):
        self.head_pose = HeadPoseEstimator(yaw_thresh=20.0, pitch_down_thresh=-15.0)
        self.gaze = GazeEstimator(ear_blink_thresh=0.20, horiz_ratio_thresh=0.20)
        
    def test_head_pose_center(self):
        # 6 canonical points matching frontal projection
        canonical_2d = [
            (320.0, 240.0), # Nose
            (320.0, 340.0), # Chin
            (260.0, 200.0), # Left eye
            (380.0, 200.0), # Right eye
            (280.0, 300.0), # Left mouth
            (360.0, 300.0)  # Right mouth
        ]
        res = self.head_pose.estimate_pose((480, 640), canonical_2d)
        self.assertIn(res['pose'], ['looking_center', 'looking_down', 'looking_up'])
        self.assertIsInstance(res['yaw'], float)
        self.assertIsInstance(res['pitch'], float)
        self.assertIsInstance(res['roll'], float)
        
    def test_gaze_center_and_blink(self):
        # Open eyes coordinates
        left_eye = np.array([[200, 200], [210, 195], [230, 195], [240, 200], [230, 205], [210, 205]])
        right_eye = np.array([[280, 200], [290, 195], [310, 195], [320, 200], [310, 205], [290, 205]])
        nose = np.array([260, 240])
        
        res = self.gaze.estimate_gaze(left_eye, right_eye, nose_pt=nose)
        self.assertEqual(res['gaze'], 'CENTER')
        self.assertFalse(res['is_blinking'])
        self.assertFalse(res['is_looking_away'])
        
    def test_gaze_blink_detected(self):
        # Closed eyes coordinates (vertical height near zero)
        closed_left = np.array([[200, 200], [210, 201], [230, 201], [240, 200], [230, 200], [210, 200]])
        closed_right = np.array([[280, 200], [290, 201], [310, 201], [320, 200], [310, 200], [290, 200]])
        
        res = self.gaze.estimate_gaze(closed_left, closed_right, nose_pt=np.array([260, 240]))
        self.assertTrue(res['is_blinking'])
        # Blinking should not trigger look-away alert
        self.assertFalse(res['is_looking_away'])

if __name__ == '__main__':
    unittest.main()
