"""Database initialization and helpers for all three databases."""
import sqlite3
import os
from config import Config


def get_db(role):
    """Get database connection by role."""
    db_map = {
        'student': Config.STUDENT_DB,
        'teacher': Config.TEACHER_DB,
        'admin': Config.ADMIN_DB
    }
    db_path = db_map.get(role)
    if not db_path:
        raise ValueError(f"Invalid role: {role}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_all_databases():
    """Initialize all three databases with their schemas."""
    Config.init()
    _init_student_db()
    _init_teacher_db()
    _init_admin_db()
    print("[DB] All databases initialized.")


def _init_student_db():
    conn = sqlite3.connect(Config.STUDENT_DB)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        uid TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT DEFAULT '',
        phone TEXT DEFAULT '',
        section TEXT DEFAULT '',
        semester TEXT DEFAULT '',
        department TEXT DEFAULT '',
        password_hash TEXT NOT NULL,
        face_embeddings BLOB,
        face_registered INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_uid TEXT NOT NULL,
        session_id TEXT NOT NULL,
        class_id TEXT DEFAULT '',
        subject TEXT DEFAULT '',
        section TEXT DEFAULT '',
        date TEXT NOT NULL,
        class_duration_min REAL DEFAULT 0,
        present_duration_min REAL DEFAULT 0,
        presence_percentage REAL DEFAULT 0,
        status TEXT DEFAULT 'absent',
        marked_by TEXT DEFAULT 'ai',
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS student_assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assignment_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        subject TEXT DEFAULT '',
        section TEXT DEFAULT '',
        file_path TEXT DEFAULT '',
        file_name TEXT DEFAULT '',
        due_date TEXT DEFAULT '',
        uploaded_by TEXT DEFAULT '',
        max_score REAL DEFAULT 100,
        attachment_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS assignment_submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assignment_id TEXT NOT NULL,
        student_uid TEXT NOT NULL,
        file_path TEXT DEFAULT '',
        file_name TEXT DEFAULT '',
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        grade TEXT DEFAULT '',
        feedback TEXT DEFAULT '',
        ai_summary TEXT DEFAULT '',
        ai_relevance_score REAL DEFAULT 0,
        ai_quality_rating REAL DEFAULT 0,
        ai_remarks TEXT DEFAULT '',
        ai_analyzed INTEGER DEFAULT 0,
        UNIQUE(assignment_id, student_uid)
    )''')

    # Migration: add AI columns to existing submissions tables
    for col in ['ai_summary', 'ai_remarks', 'ai_relevance_score', 'ai_quality_rating', 'ai_analyzed']:
        try:
            col_type = 'TEXT DEFAULT ""' if 'summary' in col or 'remarks' in col else 'REAL DEFAULT 0' if 'score' in col or 'rating' in col else 'INTEGER DEFAULT 0'
            c.execute(f'ALTER TABLE assignment_submissions ADD COLUMN {col} {col_type}')
        except: pass

    # Migration: add new columns to student_assignments
    for col in ['max_score', 'attachment_count']:
        try:
            col_type = 'REAL DEFAULT 100' if col == 'max_score' else 'INTEGER DEFAULT 0'
            c.execute(f'ALTER TABLE student_assignments ADD COLUMN {col} {col_type}')
        except: pass

    # Assignment attachments (shared — stored in student DB for student access)
    c.execute('''CREATE TABLE IF NOT EXISTS assignment_attachments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assignment_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_size INTEGER DEFAULT 0,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Student engagement tracking
    c.execute('''CREATE TABLE IF NOT EXISTS student_engagement (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_uid TEXT NOT NULL,
        session_id TEXT NOT NULL,
        attention_score REAL DEFAULT 50,
        mood TEXT DEFAULT 'neutral',
        hand_raises INTEGER DEFAULT 0,
        answers_given INTEGER DEFAULT 0,
        engagement_score REAL DEFAULT 50,
        distraction_count INTEGER DEFAULT 0,
        looking_away_count INTEGER DEFAULT 0,
        timeline_data TEXT DEFAULT '[]',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(student_uid, session_id)
    )''')

    # Exam tracking for students
    c.execute('''CREATE TABLE IF NOT EXISTS student_exams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exam_id TEXT NOT NULL,
        student_uid TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        score INTEGER DEFAULT 0,
        total INTEGER DEFAULT 0,
        percentage REAL DEFAULT 0,
        submitted_at TIMESTAMP,
        UNIQUE(exam_id, student_uid)
    )''')
    
    conn.commit()
    conn.close()


def _init_teacher_db():
    conn = sqlite3.connect(Config.TEACHER_DB)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS teachers (
        uid TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT DEFAULT '',
        phone TEXT DEFAULT '',
        department TEXT DEFAULT '',
        password_hash TEXT NOT NULL,
        face_embeddings BLOB,
        face_registered INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS timetable (
        id TEXT PRIMARY KEY,
        teacher_uid TEXT NOT NULL,
        subject TEXT NOT NULL,
        section TEXT NOT NULL,
        day_of_week TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        duration_min INTEGER DEFAULT 50,
        room TEXT DEFAULT '',
        is_recurring INTEGER DEFAULT 1,
        specific_date TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS attendance_sessions (
        id TEXT PRIMARY KEY,
        timetable_id TEXT DEFAULT '',
        teacher_uid TEXT NOT NULL,
        subject TEXT DEFAULT '',
        section TEXT DEFAULT '',
        date TEXT NOT NULL,
        scheduled_start TIMESTAMP,
        actual_start TIMESTAMP,
        actual_end TIMESTAMP,
        duration_min INTEGER DEFAULT 50,
        min_presence_percent REAL DEFAULT 70,
        total_enrolled INTEGER DEFAULT 0,
        total_present INTEGER DEFAULT 0,
        total_absent INTEGER DEFAULT 0,
        status TEXT DEFAULT 'scheduled',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS assignments (
        id TEXT PRIMARY KEY,
        teacher_uid TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        subject TEXT DEFAULT '',
        section TEXT DEFAULT '',
        file_path TEXT DEFAULT '',
        file_name TEXT DEFAULT '',
        due_date TEXT DEFAULT '',
        max_score REAL DEFAULT 100,
        attachment_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Migration: add new columns to existing assignments table
    for col in ['max_score', 'attachment_count']:
        try:
            col_type = 'REAL DEFAULT 100' if col == 'max_score' else 'INTEGER DEFAULT 0'
            c.execute(f'ALTER TABLE assignments ADD COLUMN {col} {col_type}')
        except: pass

    # Assignment attachments (teacher copy)
    c.execute('''CREATE TABLE IF NOT EXISTS assignment_attachments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assignment_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_size INTEGER DEFAULT 0,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Chat messages for AI chatbot
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_uid TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # ── Exam System Tables ──
    c.execute('''CREATE TABLE IF NOT EXISTS exams (
        id TEXT PRIMARY KEY,
        teacher_uid TEXT NOT NULL,
        title TEXT NOT NULL,
        subject TEXT DEFAULT '',
        topic TEXT DEFAULT '',
        course TEXT DEFAULT '',
        section TEXT DEFAULT '',
        difficulty TEXT DEFAULT 'medium',
        num_questions INTEGER DEFAULT 10,
        duration_min INTEGER DEFAULT 30,
        status TEXT DEFAULT 'draft',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        published_at TIMESTAMP,
        closed_at TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS exam_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exam_id TEXT NOT NULL,
        question_num INTEGER NOT NULL,
        question_text TEXT NOT NULL,
        option_a TEXT NOT NULL,
        option_b TEXT NOT NULL,
        option_c TEXT NOT NULL,
        option_d TEXT NOT NULL,
        correct_option TEXT NOT NULL,
        explanation TEXT DEFAULT '',
        UNIQUE(exam_id, question_num)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS exam_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exam_id TEXT NOT NULL,
        student_uid TEXT NOT NULL,
        student_name TEXT DEFAULT '',
        score INTEGER DEFAULT 0,
        total INTEGER DEFAULT 0,
        percentage REAL DEFAULT 0,
        answers TEXT DEFAULT '{}',
        time_taken_sec INTEGER DEFAULT 0,
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        proctor_summary TEXT DEFAULT '{}',
        head_turn_count INTEGER DEFAULT 0,
        eye_movement_alerts INTEGER DEFAULT 0,
        tab_switch_count INTEGER DEFAULT 0,
        face_not_visible_count INTEGER DEFAULT 0,
        risk_level TEXT DEFAULT 'low',
        ai_analysis TEXT DEFAULT '',
        UNIQUE(exam_id, student_uid)
    )''')
    
    conn.commit()
    conn.close()


def _init_admin_db():
    conn = sqlite3.connect(Config.ADMIN_DB)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
        uid TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT DEFAULT '',
        password_hash TEXT NOT NULL,
        face_embeddings BLOB,
        face_registered INTEGER DEFAULT 0,
        role TEXT DEFAULT 'admin',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS sections (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT DEFAULT '',
        semester TEXT DEFAULT '',
        academic_year TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_uid TEXT DEFAULT '',
        user_role TEXT DEFAULT '',
        action TEXT NOT NULL,
        details TEXT DEFAULT '',
        ip_address TEXT DEFAULT '',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Migration: add ip_address column if not exists (for existing DBs)
    try:
        c.execute('ALTER TABLE audit_log ADD COLUMN ip_address TEXT DEFAULT ""')
    except: pass  # Column already exists
    
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT DEFAULT '',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Insert default settings
    defaults = [
        ('min_presence_percent', '70'),
        ('face_threshold', '0.45'),
        ('max_login_attempts', '5'),
        ('session_timeout_hours', '24'),
    ]
    for key, val in defaults:
        c.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (key, val))
    
    conn.commit()
    conn.close()
