"""
ExamGuard AI (PRJ 137) - 3D Head Pose Estimation Engine
Calculates 3D Euler angles (Yaw, Pitch, Roll) using 2D-to-3D facial landmark correspondence via OpenCV solvePnP.
Includes temporal moving average smoothing to prevent single-frame false alarms.
"""

import cv2
import numpy as np
from collections import deque

class HeadPoseEstimator:
    def __init__(self, yaw_thresh=25.0, pitch_down_thresh=-18.0, pitch_up_thresh=20.0, roll_thresh=20.0, smoothing_window=5):
        self.yaw_thresh = yaw_thresh
        self.pitch_down_thresh = pitch_down_thresh
        self.pitch_up_thresh = pitch_up_thresh
        self.roll_thresh = roll_thresh
        
        # Temporal smoothing buffers for Euler angles
        self.yaw_history = deque(maxlen=smoothing_window)
        self.pitch_history = deque(maxlen=smoothing_window)
        self.roll_history = deque(maxlen=smoothing_window)
        
        # 3D canonical model coordinates of key facial landmarks (in mm)
        # Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left Mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)

    def estimate_pose(self, frame_shape, image_points_2d):
        """
        Calculates smoothed Euler angles from 6 facial landmarks.
        image_points_2d: 6x2 array of [(x, y)] corresponding to model_points_3d
        """
        if image_points_2d is None or len(image_points_2d) < 6:
            return {
                'pose': 'unknown',
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'is_deviated': False
            }
            
        h, w = frame_shape[:2]
        
        # Camera intrinsic matrix approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1)) # Assuming minimal lens distortion
        
        image_pts = np.array(image_points_2d, dtype=np.float64)
        
        # Solve Perspective-n-Point
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points_3d,
            image_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {
                'pose': 'unknown',
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'is_deviated': False
            }
            
        # Convert rotation vector to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        
        # Extract Euler angles (decompose projection matrix)
        proj_matrix = np.hstack((rot_mat, trans_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        raw_pitch = float(euler_angles[0][0])
        raw_yaw = float(euler_angles[1][0])
        raw_roll = float(euler_angles[2][0])
        
        # Temporal smoothing
        self.yaw_history.append(raw_yaw)
        self.pitch_history.append(raw_pitch)
        self.roll_history.append(raw_roll)
        
        yaw = float(np.mean(self.yaw_history))
        pitch = float(np.mean(self.pitch_history))
        roll = float(np.mean(self.roll_history))
        
        # Classification
        pose = 'looking_center'
        is_deviated = False
        
        if pitch < self.pitch_down_thresh:
            pose = 'looking_down'
            is_deviated = True
        elif pitch > self.pitch_up_thresh:
            pose = 'looking_up'
            is_deviated = True
        elif yaw > self.yaw_thresh:
            pose = 'looking_right'
            is_deviated = True
        elif yaw < -self.yaw_thresh:
            pose = 'looking_left'
            is_deviated = True
            
        return {
            'pose': pose,
            'yaw': round(yaw, 2),
            'pitch': round(pitch, 2),
            'roll': round(roll, 2),
            'is_deviated': is_deviated
        }
