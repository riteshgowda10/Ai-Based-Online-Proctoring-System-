"""
ExamGuard AI (PRJ 137) - Face Representation & Identity Verification Module
Uses pre-trained InceptionResnetV1 (FaceNet VGGFace2) to extract 512-D L2-normalized embeddings.
Compares exam-time candidate embeddings against enrolled identity using Cosine Similarity.
"""

import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN

class IdentityVerifier:
    def __init__(self, similarity_threshold=0.65, device=None):
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Pre-trained MTCNN for face crop alignment
        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            device=self.device
        )
        
        # Pre-trained FaceNet model on VGGFace2
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.similarity_threshold = similarity_threshold
        
        # Memory storage for enrolled embeddings {student_id: 512-D numpy vector}
        self.enrolled_embeddings = {}
        
    def extract_embedding(self, frame):
        """
        Detects face, crops and aligns to 160x160, and generates 512-D L2-normalized embedding.
        """
        if frame is None or frame.size == 0:
            return None
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # MTCNN crops and normalizes to (-1, 1) tensor of shape [3, 160, 160]
        face_tensor = self.mtcnn(rgb)
        
        if face_tensor is None:
            return None
            
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.resnet(face_tensor)
            # Normalize embedding
            embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
            
        return embedding.squeeze(0).cpu().numpy()
        
    def register_candidate(self, student_id, frame):
        """
        Enrolls a candidate's baseline reference face embedding.
        """
        emb = self.extract_embedding(frame)
        if emb is None:
            return False, "No clear face detected during identity enrollment."
            
        self.enrolled_embeddings[student_id] = emb
        return True, "Identity enrolled successfully."
        
    def verify_identity(self, student_id, current_frame):
        """
        Compares current face against enrolled candidate profile.
        Returns:
            match (bool): True if cosine similarity >= threshold
            similarity (float): Cosine similarity score [-1.0, 1.0]
            status (str): "MATCH", "MISMATCH", "NO_FACE", "NOT_ENROLLED"
        """
        if student_id not in self.enrolled_embeddings:
            # Fallback if student not pre-enrolled: enroll first verified frame
            success, msg = self.register_candidate(student_id, current_frame)
            if success:
                return True, 1.0, "MATCH (Auto-enrolled baseline)"
            return False, 0.0, "NOT_ENROLLED"
            
        current_emb = self.extract_embedding(current_frame)
        if current_emb is None:
            return False, 0.0, "NO_FACE"
            
        enrolled_emb = self.enrolled_embeddings[student_id]
        
        # Cosine similarity
        similarity = float(np.dot(enrolled_emb, current_emb) / (np.linalg.norm(enrolled_emb) * np.linalg.norm(current_emb) + 1e-8))
        match = similarity >= self.similarity_threshold
        status = "MATCH" if match else "MISMATCH"
        
        return match, round(similarity, 4), status
