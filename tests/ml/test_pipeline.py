"""
End-to-End Unit Test: Unified Real-Time Proctoring Pipeline
Verifies complete flow: Frame -> Preprocessing -> Detectors -> Event Engine -> Risk Engine -> Structured Output
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.inference.pipeline import RealtimeProctoringPipeline

class TestProctoringPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RealtimeProctoringPipeline()
        
    def test_pipeline_on_synthetic_frame(self):
        # Create a synthetic 640x640 frame
        frame = np.ones((640, 640, 3), dtype=np.uint8) * 200
        
        telemetry = {
            'tab_switches': 1,
            'copy_attempts': 0,
            'screenshot_attempts': 0
        }
        
        output = self.pipeline.process_frame(frame, student_id="STUDENT_001", browser_telemetry=telemetry)
        
        # Verify schema
        self.assertIn('timestamp', output)
        self.assertIn('frame_id', output)
        self.assertIn('face_count', output)
        self.assertIn('identity_match', output)
        self.assertIn('head_pose', output)
        self.assertIn('gaze', output)
        self.assertIn('objects', output)
        self.assertIn('events', output)
        self.assertIn('risk_indicator', output)
        self.assertIn('integrity_index', output)
        self.assertIn('risk_level', output)
        self.assertIn('latency_ms', output)
        self.assertTrue(output['latency_ms'] > 0.0)

if __name__ == '__main__':
    unittest.main()
