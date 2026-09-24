# EXAMGUARD AI — MODEL LIMITATIONS, EDGE CASES & FAIRNESS ANALYSIS

## 1. Environmental & Visual Edge Cases

### 1.1 Difficult Lighting & Low-Light Conditions
- **Issue**: Strong backlighting (e.g. window behind candidate) silhouettes the face, degrading MTCNN face detection confidence below the 0.80 threshold.
- **Mitigation**: Pre-exam environment verification check warns the student before the test starts if mean frame luminance is $< 40$ or $> 220$ on an 8-bit scale.

### 1.2 Facial Occlusion & Eyeglasses
- **Issue**: Thick-rimmed glasses or glare reflections on lenses can distort landmark detection for the eye corners, falsely reducing the Eye Aspect Ratio (EAR).
- **Mitigation**: Temporal moving average smoothing filters transient blinks and reflections; gaze deviation requires $\ge 2.5$ seconds of continuous deviation before generating an alert.

### 1.3 Off-Angle & Extreme Webcams
- **Issue**: Low-mounted laptop cameras (looking up at candidate from below) cause natural pitch angles to appear as `looking_up`.
- **Mitigation**: System supports initial calibration frame during identity onboarding to establish the candidate's neutral resting head pose.

### 1.4 Background Posters & Photos
- **Issue**: Static portrait photos, album covers, or movie posters on background walls may trigger MTCNN secondary face detections.
- **Mitigation**: The system requires multiple faces to persist for $\ge 5$ consecutive frames and filters out non-moving background regions.

---

## 2. Demographic & Skin-Tone Considerations
- Pre-trained facial representation networks (FaceNet VGGFace2) have known accuracy variances across extreme lighting and diverse skin tones.
- ExamGuard AI explicitly avoids automated failure or expulsion. All ML predictions serve solely as **configurable risk indicators** and **evidence timestamps** for human proctor review. No student is disqualified without manual administrative audit.
