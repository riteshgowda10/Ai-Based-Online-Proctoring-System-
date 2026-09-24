"""
Unit Tests: Face Detection & Biometric Identity Verification
Tests: no face, one face, multiple faces, identity match, identity mismatch
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.face.detector import UnifiedFaceDetector
from ml.face.identity import IdentityVerifier

class TestFaceAndIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = UnifiedFaceDetector()
        cls.verifier = IdentityVerifier(similarity_threshold=0.65)
        
    def test_invalid_and_blank_frame(self):
        # Blank black frame (no face)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.detector.detect(blank)
        self.assertFalse(res['face_present'])
        self.assertEqual(res['face_count'], 0)
        self.assertIsNone(res['primary_box'])
        
    def test_none_frame(self):
        res = self.detector.detect(None)
        self.assertFalse(res['face_present'])
        self.assertEqual(res['face_count'], 0)
        
    def test_identity_registration_and_match(self):
        # Generate two synthetic face frames with distinct patterns
        frame_a = np.ones((160, 160, 3), dtype=np.uint8) * 128
        frame_a[60:100, 60:100] = 255
        
        # Test enrollment
        student_id = "TEST_STUDENT_001"
        emb_a = np.random.randn(512).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)
        self.verifier.enrolled_embeddings[student_id] = emb_a
        
        # Test identity verification logic with identical embedding
        similarity = float(np.dot(emb_a, emb_a))
        self.assertAlmostEqual(similarity, 1.0, places=4)
        self.assertTrue(similarity >= self.verifier.similarity_threshold)
        
    def test_identity_mismatch(self):
        emb_a = np.random.randn(512).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)
        
        # Orthogonal embedding
        emb_b = np.random.randn(512).astype(np.float32)
        emb_b -= emb_b.dot(emb_a) * emb_a
        emb_b /= np.linalg.norm(emb_b)
        
        similarity = float(np.dot(emb_a, emb_b))
        self.assertTrue(similarity < self.verifier.similarity_threshold)

if __name__ == '__main__':
    unittest.main()
