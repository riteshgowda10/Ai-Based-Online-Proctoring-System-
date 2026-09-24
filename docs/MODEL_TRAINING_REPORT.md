# EXAMGUARD AI (PRJ 137) - ML MODEL TRAINING & BENCHMARK REPORT

Generated at: 2026-09-24 20:10:08

```
EXAMGUARD AI (PRJ 137) - MODEL TRAINING PIPELINE STARTED AT 2026-09-24 20:09:57.230277

======================================================================
TRAINING MODEL 1: MULTI-MODAL MALPRACTICE RISK CLASSIFIER
======================================================================
Training Samples: 4800 | Test Samples: 1200
Test Accuracy: 95.58%
Weighted F1-Score: 0.9557
5-Fold Cross Validation Accuracy: 95.67% (+/- 0.44%)

Classification Report:
              precision    recall  f1-score   support

     LOW (0)       0.97      0.98      0.97       734
MODERATE (1)       0.90      0.89      0.89       227
    HIGH (2)       0.99      0.94      0.96       145
CRITICAL (3)       0.96      0.99      0.97        94

    accuracy                           0.96      1200
   macro avg       0.95      0.95      0.95      1200
weighted avg       0.96      0.96      0.96      1200

Confusion Matrix:
[[716  18   0   0]
 [ 24 202   1   0]
 [  0   5 136   4]
 [  0   0   1  93]]

Feature Importance Rankings:
  - screenshot_attempts   : 15.84%
  - tab_switches          : 15.19%
  - copy_attempts         : 15.02%
  - gaze_direction        : 14.34%
  - head_pose             : 13.47%
  - head_yaw              : 10.35%
  - head_pitch            :  6.95%
  - objects_detected      :  4.02%
  - multiple_faces        :  2.04%
  - voice_detected        :  1.24%
  - mouth_moving          :  0.93%
  - face_present          :  0.60%

[OK] Model 1 saved successfully to: C:\Users\roope\.gemini\antigravity\scratch\online-exam-proctor\models\malpractice_risk_model.pkl

======================================================================
TRAINING MODEL 2: HEAD POSE & GAZE DEVIATION CLASSIFIER
======================================================================
Training Samples: 4000 | Test Samples: 1000
Test Accuracy: 99.90%
Weighted F1-Score: 0.9990
5-Fold Cross Validation Accuracy: 99.86% (+/- 0.12%)

Classification Report:
                   precision    recall  f1-score   support

       Center (0)       1.00      1.00      1.00       450
 Looking Left (1)       1.00      1.00      1.00       206
Looking Right (2)       1.00      0.99      1.00       193
 Looking Down (3)       1.00      1.00      1.00       151

         accuracy                           1.00      1000
        macro avg       1.00      1.00      1.00      1000
     weighted avg       1.00      1.00      1.00      1000

Confusion Matrix:
[[450   0   0   0]
 [  0 206   0   0]
 [  1   0 192   0]
 [  0   0   0 151]]

[OK] Model 2 saved successfully to: C:\Users\roope\.gemini\antigravity\scratch\online-exam-proctor\models\head_pose_gaze_model.pkl

======================================================================
TRAINING MODEL 3: CANDIDATE BEHAVIORAL ANOMALY DETECTOR (UNSUPERVISED)
======================================================================
Evaluated Samples: 4000
Contamination Rate: 12.0%
Anomaly Detection Accuracy vs Baseline: 99.67%
Anomaly Detection F1-Score: 0.9863
ROC-AUC (Anomaly Separation Score): 1.0000
Decision Function Score Range: [-0.231, 0.148]

[OK] Model 3 saved successfully to: C:\Users\roope\.gemini\antigravity\scratch\online-exam-proctor\models\candidate_anomaly_detector.pkl

======================================================================
MODEL TRAINING BENCHMARK SUMMARY
======================================================================
Model Name                                 | Algorithm                    | Accuracy   | F1-Score
--------------------------------------------------------------------------------------------
Multi-Modal Malpractice Risk Classifier    | RandomForestClassifier (150 trees) | 95.58%     | 0.9557
Head Pose & Gaze Deviation Classifier      | HistGradientBoosting (StandardScaler Pipeline) | 99.90%     | 0.9990
Candidate Behavioral Anomaly Detector      | IsolationForest (RobustScaler Pipeline) | 99.67%     | 0.9863

Total Pipeline Execution Time: 11.73 seconds
```
