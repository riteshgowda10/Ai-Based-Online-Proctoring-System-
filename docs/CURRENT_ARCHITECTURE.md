# EXAMGUARD AI — CURRENT ARCHITECTURE SPECIFICATION

This document outlines the current system architecture, data pipelines, event processing flows, state machines, and risk calculation formulas of **EXAMGUARD AI**.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXAMGUARD AI SYSTEM                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│     CANDIDATE UI      │                             │      PROCTOR UI       │
│  (HTML/CSS/JS/Jinja)  │                             │  (HTML/CSS/JS/Jinja)  │
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            │ HTTP / SSE / REST APIs                              │ Telemetry / Actions
            ▼                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FLASK BACKEND ENGINE                             │
│                                (app.py)                                     │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│    Auth & State Engine   │  CBT Question / Compiler │   Proctor Action API  │
└───────────┬──────────────┴────────────┬─────────────┴───────────┬───────────┘
            │                           │                         │
            ▼                           ▼                         ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│   VISION PIPELINE     │   │     AUDIO PIPELINE    │   │    SECURITY GUARD     │
│ - MTCNN (Single/Multi)│   │ - Web Audio API Level │   │ - Screenshot Intercept│
│ - MediaPipe (Gaze/EAR)│   │ - PyAudio ZCR/Energy  │   │ - Tab Switch Tracker  │
│ - 3D Head Pose (PnP)  │   │ - Voice Activity      │   │ - Fullscreen Lock     │
│ - YOLOv8n (Objects)   │   └───────────┬───────────┘   └───────────┬───────────┘
└───────────┬───────────┘               │                           │
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRECISION RISK & EVENT PROCESSOR                         │
│                                                                             │
│  Malpractice Risk Calculation Formula:                                      │
│  Score = 5 + (NoFace * 35) + (MultiFace * 45) + (Phone * 30) + (Gaze * 15) │
│          + (HeadPose * 25) + (Audio * 15) + (TabSwitches * 20)              │
│          + (CopyAttempts * 15) + (ScreenshotAttempts * 25)                  │
│                                                                             │
│  Candidate Integrity Index = max(0, 100 - Score)                            │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REPORTING & PERSISTENCE ENGINE                           │
│ - Alert Logs (logs/alerts.log)                                              │
│ - Violations JSON (reports/violations.json)                                 │
│ - Violation Images (reports/violation_captures/)                            │
│ - PDF/HTML Evaluation Reports (reports/generated/)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Exam Session State Machine

Every candidate examination session transitions strictly through the following state machine:

```
[ CREATED ] ──> [ AUTHENTICATED ] ──> [ PRECHECK ] ──> [ VERIFIED ]
                                                             │
                                                             ▼
[ SUBMITTED ] <── [ PAUSED ] <── [ ACTIVE ] <── [ READY ]
      │               │
      ▼               ▼
[ COMPLETED ]   [ TERMINATED ]
```

---

## 3. Risk Calculation Engine Formula

EXAMGUARD AI calculates a transparent multi-signal Malpractice Risk Score ($S \in [0, 99]$) and Candidate Integrity Index ($I = \max(0, 100 - S)$):

$$S = S_{\text{base}} + \sum w_i \cdot \text{signal}_i$$

Where:
- $S_{\text{base}} = 5.0$
- $w_{\text{NO\_FACE}} = +35.0$
- $w_{\text{MULTI\_FACE}} = +45.0$
- $w_{\text{PHONE\_OBJECT}} = +30.0$
- $w_{\text{GAZE\_DEVIATION}} = +15.0$
- $w_{\text{HEAD\_LOOKING\_DOWN}} = +25.0$
- $w_{\text{VOICE\_DETECTED}} = +15.0$
- $w_{\text{TAB\_SWITCH}} = +20.0 \times \text{count}$
- $w_{\text{COPY\_ATTEMPT}} = +15.0 \times \text{count}$
- $w_{\text{SCREENSHOT\_ATTEMPT}} = +25.0 \times \text{count}$

### Risk Tier Tiers:
- **0 – 19**: `LOW` (High Integrity $\ge 81\%$)
- **20 – 49**: `MODERATE` (Moderate Integrity $51\% - 80\%$)
- **50 – 74**: `HIGH` (Low Integrity $26\% - 50\%$)
- **75 – 99**: `CRITICAL` (Critical Integrity $\le 25\%$)

---

## 4. Hardware Verification & Identity Pipeline (`/instructions`)

Candidate onboarding enforces 4 mandatory hardware checks prior to starting an exam session:
1. 📷 **Camera Stream Test**: Requests webcam stream and verifies face detection readiness.
2. 🎤 **Microphone Audio Test**: Web Audio API visualizer verifies live audio input volume level.
3. 🖥️ **Fullscreen Lockdown Verification**: Verifies browser capability to enter and hold fullscreen lock.
4. 📝 **Compliance Agreement**: Candidate explicitly accepts malpractice monitoring terms.

The "Start Examination" button remains disabled until all 4 verifications pass.
