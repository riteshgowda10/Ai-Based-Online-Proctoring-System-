# EXAMGUARD AI — REAL-TIME INFERENCE & INTEGRATION GUIDE

## 1. Programmatic Python API
```python
import cv2
from ml.inference.pipeline import RealtimeProctoringPipeline

# 1. Initialize complete pipeline
pipeline = RealtimeProctoringPipeline()

# 2. Capture video frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 3. Process frame with candidate context
telemetry = {
    'tab_switches': 0,
    'copy_attempts': 0,
    'screenshot_attempts': 0
}

result = pipeline.process_frame(frame, student_id="STUDENT_001", browser_telemetry=telemetry)

print(f"Risk Indicator: {result['risk_indicator']} / 100")
print(f"Risk Level: {result['risk_level']}")
print(f"Detected Events: {result['events']}")
print(f"Latency: {result['latency_ms']} ms")
```

## 2. REST API Integration (`src/dashboard/app.py`)

### Start Proctoring Session
```http
POST /api/proctoring/start
Content-Type: application/json
```

### Stop Proctoring Session
```http
POST /api/proctoring/stop
Content-Type: application/json
```

### Submit Frame for Real-Time Analysis
```http
POST /api/proctoring/frame
Content-Type: application/json

{
  "student_id": "STUDENT_001",
  "image_base64": "data:image/jpeg;base64,...",
  "telemetry": {
    "tab_switches": 1,
    "copy_attempts": 0,
    "screenshot_attempts": 0
  }
}
```

### Fetch Emitted Events Log
```http
GET /api/proctoring/events
```

### Check Proctoring Status
```http
GET /api/proctoring/status
```
