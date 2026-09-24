"""
ExamGuard AI (PRJ 137) - Telemetry & Synthetic Dataset Generator
Generates realistic training and validation datasets for the 3 Machine Learning models:
1. Multi-Modal Malpractice Risk Classification Dataset
2. Head Pose & Gaze Deviation Classification Dataset
3. Candidate Behavioral Anomaly Detection Dataset
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_malpractice_dataset(n_samples=5000):
    """
    Generates multi-modal telemetry dataset with 12 input features and target risk labels:
    0: LOW, 1: MODERATE, 2: HIGH, 3: CRITICAL
    """
    records = []
    
    for _ in range(n_samples):
        # 60% normal behavior, 20% moderate distractions, 12% high risk, 8% critical malpractice
        archetype = np.random.choice(['normal', 'distracted', 'suspicious', 'critical'], p=[0.60, 0.20, 0.12, 0.08])
        
        if archetype == 'normal':
            face_present = 1
            multiple_faces = 0
            objects_detected = 0
            gaze_direction = 0  # Center
            head_pose = 0       # Center
            head_yaw = np.random.normal(0, 4.0)
            head_pitch = np.random.normal(-2, 3.0)
            mouth_moving = int(np.random.rand() < 0.05)
            voice_detected = int(np.random.rand() < 0.03)
            tab_switches = 0 if np.random.rand() > 0.05 else 1
            copy_attempts = 0
            screenshot_attempts = 0
            risk_label = 0      # LOW
            
        elif archetype == 'distracted':
            face_present = 1
            multiple_faces = 0
            objects_detected = 0
            # Looking around or natural slight gaze deviation
            gaze_direction = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
            head_pose = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
            head_yaw = np.random.normal(12 if head_pose == 1 else (-12 if head_pose == 2 else 0), 6.0)
            head_pitch = np.random.normal(-5, 5.0)
            mouth_moving = int(np.random.rand() < 0.20)
            voice_detected = int(np.random.rand() < 0.12)
            tab_switches = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
            copy_attempts = 0
            screenshot_attempts = 0
            risk_label = 1      # MODERATE
            
        elif archetype == 'suspicious':
            face_present = int(np.random.rand() > 0.15)
            multiple_faces = int(np.random.rand() < 0.25)
            objects_detected = int(np.random.rand() < 0.40)  # phone/notes glimpse
            gaze_direction = np.random.choice([1, 2, 3], p=[0.35, 0.35, 0.30])
            head_pose = 3 if gaze_direction == 3 else np.random.choice([1, 2], p=[0.5, 0.5])
            head_yaw = np.random.normal(24 if head_pose == 1 else -24, 7.0)
            head_pitch = np.random.normal(-28 if head_pose == 3 else -8, 6.0)
            mouth_moving = int(np.random.rand() < 0.45)
            voice_detected = int(np.random.rand() < 0.35)
            tab_switches = np.random.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.25, 0.1])
            copy_attempts = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
            screenshot_attempts = int(np.random.rand() < 0.20)
            risk_label = 2      # HIGH
            
        else:  # critical
            face_present = int(np.random.rand() > 0.40)
            multiple_faces = int(np.random.rand() < 0.70)
            objects_detected = int(np.random.rand() < 0.75)  # clear phone or forbidden item
            gaze_direction = np.random.choice([1, 2, 3], p=[0.25, 0.25, 0.50])
            head_pose = 3 if gaze_direction == 3 else np.random.choice([1, 2, 4], p=[0.4, 0.4, 0.2])
            head_yaw = np.random.normal(32 if head_pose == 1 else -32, 9.0)
            head_pitch = np.random.normal(-35 if head_pose == 3 else -10, 8.0)
            mouth_moving = int(np.random.rand() < 0.75)
            voice_detected = int(np.random.rand() < 0.65)
            tab_switches = np.random.randint(2, 8)
            copy_attempts = np.random.randint(1, 6)
            screenshot_attempts = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            risk_label = 3      # CRITICAL
            
        records.append({
            'face_present': face_present,
            'multiple_faces': multiple_faces,
            'objects_detected': objects_detected,
            'gaze_direction': gaze_direction,
            'head_pose': head_pose,
            'head_yaw': round(head_yaw, 2),
            'head_pitch': round(head_pitch, 2),
            'mouth_moving': mouth_moving,
            'voice_detected': voice_detected,
            'tab_switches': tab_switches,
            'copy_attempts': copy_attempts,
            'screenshot_attempts': screenshot_attempts,
            'risk_level': risk_label
        })
        
    return pd.DataFrame(records)


def generate_head_pose_gaze_dataset(n_samples=4000):
    """
    Generates facial landmark geometry & eye coordinate dataset:
    Target Attention / Gaze Classes:
    0: Center (Focused)
    1: Looking Left
    2: Looking Right
    3: Looking Down (Phone / Desk Notes Risk)
    """
    records = []
    
    for _ in range(n_samples):
        target = np.random.choice([0, 1, 2, 3], p=[0.45, 0.20, 0.20, 0.15])
        
        if target == 0:  # Center
            nose_x = np.random.normal(0.0, 4.0)
            nose_y = np.random.normal(0.0, 3.0)
            ear_left = np.random.normal(0.32, 0.025)
            ear_right = np.random.normal(0.32, 0.025)
            eye_diff_horiz = np.random.normal(0.0, 4.5)
            eye_diff_vert = np.random.normal(0.0, 3.0)
            mouth_open = np.random.normal(0.012, 0.005)
            face_ratio = np.random.normal(1.30, 0.06)
            
        elif target == 1:  # Looking Left
            nose_x = np.random.normal(-24.0, 6.0)
            nose_y = np.random.normal(-2.0, 4.0)
            ear_left = np.random.normal(0.33, 0.03)
            ear_right = np.random.normal(0.28, 0.03)  # perspective foreshortening
            eye_diff_horiz = np.random.normal(-22.0, 5.0)
            eye_diff_vert = np.random.normal(2.0, 4.0)
            mouth_open = np.random.normal(0.015, 0.006)
            face_ratio = np.random.normal(1.20, 0.07)
            
        elif target == 2:  # Looking Right
            nose_x = np.random.normal(24.0, 6.0)
            nose_y = np.random.normal(-2.0, 4.0)
            ear_left = np.random.normal(0.28, 0.03)
            ear_right = np.random.normal(0.33, 0.03)
            eye_diff_horiz = np.random.normal(22.0, 5.0)
            eye_diff_vert = np.random.normal(2.0, 4.0)
            mouth_open = np.random.normal(0.015, 0.006)
            face_ratio = np.random.normal(1.20, 0.07)
            
        else:  # Looking Down
            nose_x = np.random.normal(0.0, 6.0)
            nose_y = np.random.normal(-26.0, 6.5)
            ear_left = np.random.normal(0.22, 0.03)   # eyelids lowered
            ear_right = np.random.normal(0.22, 0.03)
            eye_diff_horiz = np.random.normal(0.0, 6.0)
            eye_diff_vert = np.random.normal(-18.0, 5.0)
            mouth_open = np.random.normal(0.018, 0.008)
            face_ratio = np.random.normal(1.10, 0.08)
            
        records.append({
            'nose_x_offset': round(nose_x, 3),
            'nose_y_offset': round(nose_y, 3),
            'ear_left': round(max(0.05, ear_left), 3),
            'ear_right': round(max(0.05, ear_right), 3),
            'eye_diff_horiz': round(eye_diff_horiz, 3),
            'eye_diff_vert': round(eye_diff_vert, 3),
            'mouth_open_ratio': round(max(0.0, mouth_open), 4),
            'face_width_to_height_ratio': round(face_ratio, 3),
            'attention_state': target
        })
        
    return pd.DataFrame(records)


def generate_candidate_behavior_dataset(n_samples=3500):
    """
    Generates behavioral cadence time-series features for Anomaly Detection (Isolation Forest).
    Unsupervised feature space:
    - keystroke_velocity_wpm (typing cadence)
    - tab_switch_frequency (per 10 minutes)
    - gaze_jitter_rate (direction shifts per minute)
    - head_movement_variance (std dev of head yaw/pitch)
    - audio_energy_mean (mean room sound)
    - focus_stability_score (0-100)
    """
    records = []
    
    for _ in range(n_samples):
        is_normal = np.random.rand() < 0.88  # 88% normal candidates, 12% anomalous outliers
        
        if is_normal:
            wpm = np.random.normal(48, 12)
            tab_freq = np.random.exponential(0.3)
            gaze_jitter = np.random.normal(2.5, 1.2)
            head_var = np.random.normal(3.8, 1.4)
            audio_energy = np.random.exponential(0.0006)
            focus_stability = np.random.normal(92, 5.0)
            label = 1  # Inlier (Normal)
        else:
            # Anomalies: robotic pasting (0 wpm or 200 wpm), crazy head shaking, extreme sound, constant tab-switching
            anomaly_type = np.random.choice(['paste_burst', 'erratic_head', 'audio_spikes', 'tab_spammer'])
            if anomaly_type == 'paste_burst':
                wpm = np.random.choice([np.random.normal(5, 2), np.random.normal(160, 20)])
                tab_freq = np.random.uniform(4, 15)
                gaze_jitter = np.random.normal(6.0, 2.0)
                head_var = np.random.normal(5.0, 2.0)
                audio_energy = np.random.exponential(0.001)
                focus_stability = np.random.normal(45, 15)
            elif anomaly_type == 'erratic_head':
                wpm = np.random.normal(30, 15)
                tab_freq = np.random.uniform(1, 5)
                gaze_jitter = np.random.normal(14.0, 3.5)
                head_var = np.random.normal(22.0, 5.0)
                audio_energy = np.random.exponential(0.002)
                focus_stability = np.random.normal(30, 12)
            elif anomaly_type == 'audio_spikes':
                wpm = np.random.normal(40, 10)
                tab_freq = np.random.uniform(0, 3)
                gaze_jitter = np.random.normal(4.0, 2.0)
                head_var = np.random.normal(6.0, 2.0)
                audio_energy = np.random.normal(0.008, 0.002)
                focus_stability = np.random.normal(55, 10)
            else:
                wpm = np.random.normal(20, 10)
                tab_freq = np.random.uniform(12, 30)
                gaze_jitter = np.random.normal(11.0, 3.0)
                head_var = np.random.normal(12.0, 3.0)
                audio_energy = np.random.exponential(0.002)
                focus_stability = np.random.normal(25, 10)
            label = -1  # Outlier (Anomaly)
            
        records.append({
            'keystroke_velocity_wpm': max(0.0, round(wpm, 1)),
            'tab_switch_frequency': max(0.0, round(tab_freq, 2)),
            'gaze_jitter_rate': max(0.0, round(gaze_jitter, 2)),
            'head_movement_variance': max(0.1, round(head_var, 2)),
            'audio_energy_mean': max(0.00001, round(audio_energy, 6)),
            'focus_stability_score': min(100.0, max(0.0, round(focus_stability, 1))),
            'ground_truth_label': label
        })
        
    return pd.DataFrame(records)


if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(data_dir, exist_ok=True)
    
    df1 = generate_malpractice_dataset(5000)
    df1.to_csv(os.path.join(data_dir, 'malpractice_telemetry.csv'), index=False)
    print(f"Generated Malpractice Dataset: {df1.shape}")
    
    df2 = generate_head_pose_gaze_dataset(4000)
    df2.to_csv(os.path.join(data_dir, 'head_pose_gaze.csv'), index=False)
    print(f"Generated Head Pose & Gaze Dataset: {df2.shape}")
    
    df3 = generate_candidate_behavior_dataset(3500)
    df3.to_csv(os.path.join(data_dir, 'candidate_behavior.csv'), index=False)
    print(f"Generated Candidate Behavior Dataset: {df3.shape}")
