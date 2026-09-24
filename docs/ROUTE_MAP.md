# EXAMGUARD AI — ROUTE MAP & COMPATIBILITY SPECIFICATION

This document details all HTTP endpoints, templates, backend handlers, authentication rules, and backward-compatibility aliases in **EXAMGUARD AI**.

---

## 1. Primary Canonical & Legacy Compatibility Route Map

| Canonical Route | Legacy Compatibility Route (`svg`) | HTTP Method | Template / Handler | Access Level | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/` or `/login` | `/svg` | `GET` | `login.html` | Public | Dual authentication gateway for Student Candidates & Admin Proctors, plus Student Registration modal. |
| `/instructions` | `/instructionssvg` | `GET` | `instructions.html` | Candidate | Pre-exam onboarding setup, rules review, and 4-step hardware permission checks (Camera, Mic, Fullscreen, Agreement). |
| `/exam` | `/examsvg` | `GET` | `exam.html` | Candidate | Computer-Based Test workspace with MCQs, multi-language code runner, anti-screenshot defense, and live toast warnings. |
| `/dashboard` | `/dashboardsvg` | `GET` | `dashboard.html` | Proctor / Admin | Real-time proctor control dashboard displaying candidate video stream, risk index, telemetry metrics, and remote intervention controls. |

---

## 2. API Endpoints Map

### A. Authentication & Registration
- `POST /api/login`
  - **Payload**: `{"role": "student" | "admin", "username": "...", "password": "..."}`
  - **Response**: `{"status": "success", "redirect": "/instructions" | "/dashboard"}`
- `POST /api/register`
  - **Payload**: `{"name": "...", "id": "...", "course": "...", "password": "..."}`
  - **Response**: `{"status": "success", "redirect": "/instructions"}`

### B. Examination Engine
- `POST /api/run_code`
  - **Payload**: `{"lang": "python"|"javascript"|"cpp"|"java", "code": "...", "problem_id": 4}`
  - **Response**: `{"status": "success", "output": "..."}`
- `POST /api/log_violation`
  - **Payload**: `{"type": "...", "message": "..."}`
  - **Response**: `{"status": "logged", "entry": "..."}`

### C. AI Vision & Audio Telemetry
- `GET /video_feed`
  - **Response**: MJPEG Stream (`multipart/x-mixed-replace`) carrying OpenCV frames with face detection, head pose, and multi-person bounding overlays.
- `GET /api/stats`
  - **Response**: JSON telemetry payload containing integrity index, risk score, head pose angles, gaze direction, face count, tab switches, and proctor message status.
- `GET /api/alerts`
  - **Response**: JSON list of recent alert log entries from `logs/alerts.log`.

### D. Proctor Remote Control & Interventions
- `POST /api/proctor/send_message`
  - **Payload**: `{"message": "Please look straight at your screen!"}`
  - **Response**: `{"status": "success", "message": "..."}`
- `POST /api/proctor/pause_session`
  - **Payload**: `{"action": "pause" | "resume"}`
  - **Response**: `{"status": "success", "session_status": "PAUSED" | "ACTIVE"}`
- `POST /api/proctor/terminate_session`
  - **Payload**: `{}`
  - **Response**: `{"status": "success", "session_status": "TERMINATED"}`

### E. Reporting & Evidence Export
- `POST /api/generate_report`
  - **Response**: `{"status": "success", "filename": "...", "download_url": "/reports/..."}`
- `GET /reports/<filename>`
  - **Response**: File download stream (PDF or fallback HTML report).

---

## 3. Backward Compatibility Implementation Rule

To ensure existing tools or links using the `svg` suffixes continue to work seamlessly without breaking changes:

```python
# Flask Route Alias Mapping
@app.route('/svg')
def legacy_login():
    return redirect(url_for('login'))

@app.route('/instructionssvg')
def legacy_instructions():
    return redirect(url_for('instructions'))

@app.route('/examsvg')
def legacy_exam():
    return redirect(url_for('exam'))

@app.route('/dashboardsvg')
def legacy_dashboard():
    return redirect(url_for('dashboard'))
```
