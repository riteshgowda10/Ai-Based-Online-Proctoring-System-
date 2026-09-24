-- ExamGuard AI (PRJ 137) - Database Schema
-- Compatible with SQLite3, PostgreSQL, and MySQL

-- 1. Candidates Table
CREATE TABLE IF NOT EXISTS candidates (
    id VARCHAR(64) PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    course VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    face_embedding_json TEXT, -- 512-D enrolled biometric embedding vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(32) DEFAULT 'ACTIVE'
);

-- 2. Examination Sessions Table
CREATE TABLE IF NOT EXISTS exam_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    candidate_id VARCHAR(64) NOT NULL,
    exam_title VARCHAR(255) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    session_status VARCHAR(32) DEFAULT 'ACTIVE', -- ACTIVE, PAUSED, TERMINATED, COMPLETED
    candidate_integrity_index INT DEFAULT 100,
    proctoring_risk_indicator INT DEFAULT 0,
    risk_level VARCHAR(32) DEFAULT 'LOW',
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
);

-- 3. Telemetry Logs Table
CREATE TABLE IF NOT EXISTS telemetry_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(64) NOT NULL,
    frame_id BIGINT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_count INT DEFAULT 1,
    identity_match BOOLEAN DEFAULT TRUE,
    identity_similarity FLOAT DEFAULT 1.0,
    head_pose_yaw FLOAT DEFAULT 0.0,
    head_pose_pitch FLOAT DEFAULT 0.0,
    head_pose_roll FLOAT DEFAULT 0.0,
    gaze_direction VARCHAR(32) DEFAULT 'CENTER',
    objects_detected TEXT, -- JSON array of detected object labels
    risk_indicator INT DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(session_id) ON DELETE CASCADE
);

-- 4. Proctoring Events & Malpractice Table
CREATE TABLE IF NOT EXISTS proctoring_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL, -- FACE_MISSING, MULTIPLE_FACES, PHONE_DETECTED, etc.
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT DEFAULT 0.0,
    duration FLOAT DEFAULT 0.0,
    source_model VARCHAR(64) NOT NULL,
    metadata_json TEXT,
    screenshot_evidence_path VARCHAR(512),
    FOREIGN KEY (session_id) REFERENCES exam_sessions(session_id) ON DELETE CASCADE
);

-- 5. System Audit Logs Table
CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type VARCHAR(64) NOT NULL, -- LOGIN, BROADCAST, PAUSE, TERMINATE, REPORT_GENERATE
    actor_id VARCHAR(64) NOT NULL,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Initial default administrator seed
INSERT OR IGNORE INTO candidates (id, full_name, email, course, password_hash, status)
VALUES ('STUDENT_001', 'John Doe', 'student@examguard.ai', 'Computer Science & Engineering', 'scrypt:32768:8:1$placeholder', 'ACTIVE');
