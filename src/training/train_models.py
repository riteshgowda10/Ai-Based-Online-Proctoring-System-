"""
ExamGuard AI (PRJ 137) - Machine Learning Training Pipeline
Trains, benchmarks, evaluates, and serializes the 3 Core ML Models:

Model 1: Multi-Modal Malpractice Risk Classifier (Random Forest / Gradient Boosted)
Model 2: Head Pose & Gaze Deviation Classifier (Landmark Geometric Classifier)
Model 3: Candidate Behavioral Anomaly Detector (Isolation Forest Unsupervised)

Outputs:
- models/malpractice_risk_model.pkl
- models/head_pose_gaze_model.pkl
- models/candidate_anomaly_detector.pkl
- docs/MODEL_TRAINING_REPORT.md
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.dataset_generator import (
    generate_malpractice_dataset,
    generate_head_pose_gaze_dataset,
    generate_candidate_behavior_dataset
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

report_lines = []

def log_report(line=""):
    print(line)
    report_lines.append(line)

# ==============================================================================
# MODEL 1: MULTI-MODAL MALPRACTICE RISK CLASSIFIER
# ==============================================================================
def train_malpractice_risk_model():
    log_report("\n" + "="*70)
    log_report("TRAINING MODEL 1: MULTI-MODAL MALPRACTICE RISK CLASSIFIER")
    log_report("="*70)
    
    df = generate_malpractice_dataset(n_samples=6000)
    
    features = [
        'face_present', 'multiple_faces', 'objects_detected', 'gaze_direction',
        'head_pose', 'head_yaw', 'head_pitch', 'mouth_moving', 'voice_detected',
        'tab_switches', 'copy_attempts', 'screenshot_attempts'
    ]
    target = 'risk_level'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Random Forest Classifier with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    
    class_names = ['LOW (0)', 'MODERATE (1)', 'HIGH (2)', 'CRITICAL (3)']
    cls_report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)
    
    log_report(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    log_report(f"Test Accuracy: {acc*100:.2f}%")
    log_report(f"Weighted F1-Score: {f1:.4f}")
    log_report(f"5-Fold Cross Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    log_report("\nClassification Report:\n" + cls_report)
    log_report("Confusion Matrix:\n" + str(cm))
    
    # Feature Importances
    importances = rf.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    log_report("\nFeature Importance Rankings:")
    for feat, imp in feat_imp:
        log_report(f"  - {feat:22s}: {imp*100:5.2f}%")
        
    model_payload = {
        'model': rf,
        'features': features,
        'classes': ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'],
        'accuracy': float(acc),
        'f1_score': float(f1),
        'trained_at': datetime.now().isoformat()
    }
    
    save_path = os.path.join(MODELS_DIR, 'malpractice_risk_model.pkl')
    joblib.dump(model_payload, save_path)
    log_report(f"\n[OK] Model 1 saved successfully to: {save_path}")
    
    return {
        'name': 'Multi-Modal Malpractice Risk Classifier',
        'type': 'RandomForestClassifier (150 trees)',
        'accuracy': f"{acc*100:.2f}%",
        'f1': f"{f1:.4f}",
        'cv': f"{cv_scores.mean()*100:.2f}%",
        'path': 'models/malpractice_risk_model.pkl'
    }

# ==============================================================================
# MODEL 2: HEAD POSE & GAZE DEVIATION CLASSIFIER
# ==============================================================================
def train_head_pose_gaze_model():
    log_report("\n" + "="*70)
    log_report("TRAINING MODEL 2: HEAD POSE & GAZE DEVIATION CLASSIFIER")
    log_report("="*70)
    
    df = generate_head_pose_gaze_dataset(n_samples=5000)
    
    features = [
        'nose_x_offset', 'nose_y_offset', 'ear_left', 'ear_right',
        'eye_diff_horiz', 'eye_diff_vert', 'mouth_open_ratio',
        'face_width_to_height_ratio'
    ]
    target = 'attention_state'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    
    class_names = ['Center (0)', 'Looking Left (1)', 'Looking Right (2)', 'Looking Down (3)']
    cls_report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)
    
    log_report(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    log_report(f"Test Accuracy: {acc*100:.2f}%")
    log_report(f"Weighted F1-Score: {f1:.4f}")
    log_report(f"5-Fold Cross Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    log_report("\nClassification Report:\n" + cls_report)
    log_report("Confusion Matrix:\n" + str(cm))
    
    model_payload = {
        'pipeline': pipeline,
        'features': features,
        'classes': ['Center', 'Turned Left', 'Turned Right', 'Looking Down (Phone Risk)'],
        'accuracy': float(acc),
        'f1_score': float(f1),
        'trained_at': datetime.now().isoformat()
    }
    
    save_path = os.path.join(MODELS_DIR, 'head_pose_gaze_model.pkl')
    joblib.dump(model_payload, save_path)
    log_report(f"\n[OK] Model 2 saved successfully to: {save_path}")
    
    return {
        'name': 'Head Pose & Gaze Deviation Classifier',
        'type': 'HistGradientBoosting (StandardScaler Pipeline)',
        'accuracy': f"{acc*100:.2f}%",
        'f1': f"{f1:.4f}",
        'cv': f"{cv_scores.mean()*100:.2f}%",
        'path': 'models/head_pose_gaze_model.pkl'
    }

# ==============================================================================
# MODEL 3: CANDIDATE BEHAVIORAL ANOMALY DETECTOR
# ==============================================================================
def train_candidate_anomaly_detector():
    log_report("\n" + "="*70)
    log_report("TRAINING MODEL 3: CANDIDATE BEHAVIORAL ANOMALY DETECTOR (UNSUPERVISED)")
    log_report("="*70)
    
    df = generate_candidate_behavior_dataset(n_samples=4000)
    
    features = [
        'keystroke_velocity_wpm', 'tab_switch_frequency', 'gaze_jitter_rate',
        'head_movement_variance', 'audio_energy_mean', 'focus_stability_score'
    ]
    
    X = df[features]
    y_ground_truth = df['ground_truth_label']  # 1 = Normal, -1 = Outlier
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest for Unsupervised Anomaly Detection
    contamination_rate = 0.12  # Expected 12% anomalous exam behavior
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination_rate,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    iso_forest.fit(X_scaled)
    predictions = iso_forest.predict(X_scaled)  # 1: Inlier, -1: Outlier
    scores = iso_forest.decision_function(X_scaled)
    
    # Benchmark against ground truth labels
    acc = accuracy_score(y_ground_truth, predictions)
    f1 = f1_score(y_ground_truth, predictions, pos_label=-1)
    roc_auc = roc_auc_score(y_ground_truth == -1, -scores)
    
    log_report(f"Evaluated Samples: {len(X)}")
    log_report(f"Contamination Rate: {contamination_rate*100:.1f}%")
    log_report(f"Anomaly Detection Accuracy vs Baseline: {acc*100:.2f}%")
    log_report(f"Anomaly Detection F1-Score: {f1:.4f}")
    log_report(f"ROC-AUC (Anomaly Separation Score): {roc_auc:.4f}")
    log_report(f"Decision Function Score Range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    model_payload = {
        'model': iso_forest,
        'scaler': scaler,
        'features': features,
        'contamination': contamination_rate,
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'trained_at': datetime.now().isoformat()
    }
    
    save_path = os.path.join(MODELS_DIR, 'candidate_anomaly_detector.pkl')
    joblib.dump(model_payload, save_path)
    log_report(f"\n[OK] Model 3 saved successfully to: {save_path}")
    
    return {
        'name': 'Candidate Behavioral Anomaly Detector',
        'type': 'IsolationForest (RobustScaler Pipeline)',
        'accuracy': f"{acc*100:.2f}%",
        'f1': f"{f1:.4f}",
        'cv': f"ROC-AUC: {roc_auc:.4f}",
        'path': 'models/candidate_anomaly_detector.pkl'
    }

# ==============================================================================
# MAIN EXECUTION & SUMMARY GENERATION
# ==============================================================================
if __name__ == '__main__':
    start_time = datetime.now()
    log_report(f"EXAMGUARD AI (PRJ 137) - MODEL TRAINING PIPELINE STARTED AT {start_time}")
    
    m1_res = train_malpractice_risk_model()
    m2_res = train_head_pose_gaze_model()
    m3_res = train_candidate_anomaly_detector()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log_report("\n" + "="*70)
    log_report("MODEL TRAINING BENCHMARK SUMMARY")
    log_report("="*70)
    log_report(f"{'Model Name':<42} | {'Algorithm':<28} | {'Accuracy':<10} | {'F1-Score'}")
    log_report("-" * 92)
    for res in [m1_res, m2_res, m3_res]:
        log_report(f"{res['name']:<42} | {res['type']:<28} | {res['accuracy']:<10} | {res['f1']}")
    log_report(f"\nTotal Pipeline Execution Time: {duration:.2f} seconds")
    
    # Save training report markdown
    report_md_path = os.path.join(DOCS_DIR, 'MODEL_TRAINING_REPORT.md')
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write("# EXAMGUARD AI (PRJ 137) - ML MODEL TRAINING & BENCHMARK REPORT\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("```\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")
    print(f"\nTraining Report written to: {report_md_path}")
