import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'attendai-super-secret-key-2026')
    JWT_SECRET = os.environ.get('JWT_SECRET', 'attendai-jwt-secret-key-2026')
    JWT_EXPIRY_HOURS = 24
    
    # Separate databases for privacy
    DB_DIR = os.path.join(BASE_DIR, 'databases')
    STUDENT_DB = os.path.join(DB_DIR, 'students.db')
    TEACHER_DB = os.path.join(DB_DIR, 'teachers.db')
    ADMIN_DB = os.path.join(DB_DIR, 'admin.db')
    
    # Upload directory for assignments
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # Face recognition settings (OpenCV YuNet + SFace)
    FACE_SIMILARITY_THRESHOLD = 0.4  # SFace cosine similarity threshold
    FACE_DET_SIZE = (640, 640)
    MIN_PRESENCE_PERCENT = 70
    FRAME_SKIP = 3  # Process every Nth frame
    RECHECK_INTERVAL = 30  # Seconds between re-verification
    
    # Face engine
    FACE_ENGINE = 'mediapipe'  # mediapipe (default) or insightface
    
    # Groq AI Configuration
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')  # Set your Groq API key as environment variable
    GROQ_MODEL = 'llama-3.3-70b-versatile'
    
    DATABASE_DIR = DB_DIR  # Alias for backward compatibility
    
    @staticmethod
    def init():
        """Create required directories."""
        os.makedirs(Config.DB_DIR, exist_ok=True)
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.UPLOAD_DIR, 'assignments'), exist_ok=True)
        os.makedirs(os.path.join(Config.UPLOAD_DIR, 'submissions'), exist_ok=True)
        os.makedirs(os.path.join(Config.UPLOAD_DIR, 'profiles'), exist_ok=True)
