# ExamGuard AI — Database Specification & Setup

## 1. Overview
ExamGuard AI uses a lightweight, transactional relational database schema to store candidate registrations, biometric face embeddings, exam session metrics, time-series telemetry logs, and proctoring audit trails.

## 2. Default Engine
- **Development / Standalone**: SQLite3 (`database/proctoring.db`)
- **Production Option**: PostgreSQL 14+ / MySQL 8.0+

## 3. Schema Architecture
- `candidates`: Stores candidate profile, academic course, authentication hash, and enrolled 512-D FaceNet embeddings.
- `exam_sessions`: Manages candidate session states (`ACTIVE`, `PAUSED`, `TERMINATED`, `COMPLETED`), current integrity index, and risk indicator.
- `telemetry_logs`: High-frequency frame telemetry capturing head pose, gaze, face counts, and detected objects.
- `proctoring_events`: Debounced suspicious event entries with confidence, duration, and screenshot evidence paths.
- `audit_logs`: Administrative actions (session pauses, terminations, proctor broadcasts).

## 4. Setup Instructions
### SQLite (Default)
```powershell
# Initialize local SQLite database
sqlite3 database/proctoring.db < database/schema.sql
```

### PostgreSQL Migration
```powershell
psql -U postgres -d examguard_db -f database/schema.sql
```

## 5. Environment Variables
Configure via `.env`:
```
DATABASE_URL=sqlite:///database/proctoring.db
# Or PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/examguard_db
```
