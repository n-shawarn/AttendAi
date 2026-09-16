"""AttendAI — Flask Backend Server."""
import os, sys, uuid, json, base64, hashlib, threading, time
from datetime import datetime, timedelta
from functools import wraps

import bcrypt, jwt, numpy as np
import eventlet
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from config import Config
from models import init_all_databases, get_db

# AI & Document parsing
from groq import Groq
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import docx
except ImportError:
    docx = None

groq_client = None
def get_groq():
    global groq_client
    if groq_client is None:
        groq_client = Groq(api_key=Config.GROQ_API_KEY)
    return groq_client

# ── App Setup ──────────────────────────────────────────────
app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Init databases on startup
Config.init()
init_all_databases()

# ── Face Engine (OpenCV DNN: YuNet + SFace) ────────────────
import cv2

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'face_models')
YUNET_MODEL = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL = os.path.join(MODEL_DIR, 'face_recognition_sface_2021dec.onnx')

face_detector_instance = None
face_recognizer_instance = None
EMBEDDING_DIM = 128  # SFace output dimension

def get_face_detector(img_w=320, img_h=320):
    """Lazy-load YuNet face detector."""
    global face_detector_instance
    if face_detector_instance is None:
        if not os.path.exists(YUNET_MODEL):
            print(f"[FACE] ERROR: YuNet model not found at {YUNET_MODEL}")
            return None
        face_detector_instance = cv2.FaceDetectorYN.create(
            YUNET_MODEL, '', (img_w, img_h),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        print("[FACE] [OK] YuNet face detector loaded.")
    return face_detector_instance

def get_face_recognizer():
    """Lazy-load SFace recognizer."""
    global face_recognizer_instance
    if face_recognizer_instance is None:
        if not os.path.exists(SFACE_MODEL):
            print(f"[FACE] ERROR: SFace model not found at {SFACE_MODEL}")
            return None
        face_recognizer_instance = cv2.FaceRecognizerSF.create(SFACE_MODEL, '')
        print("[FACE] [OK] SFace face recognizer loaded.")
    return face_recognizer_instance

def detect_faces_mediapipe(img):
    """Detect faces using YuNet and extract 128-D embeddings using SFace.
    Returns list of dicts with 'bbox', 'embedding', 'confidence'."""
    h, w = img.shape[:2]

    detector = get_face_detector(w, h)
    recognizer = get_face_recognizer()
    if detector is None or recognizer is None:
        return []

    # Update detector input size to match image
    detector.setInputSize((w, h))
    _, raw_faces = detector.detect(img)

    faces = []
    if raw_faces is not None:
        for face_info in raw_faces:
            # YuNet returns [x, y, w, h, ..., score]
            x1 = max(0, int(face_info[0]))
            y1 = max(0, int(face_info[1]))
            fw = int(face_info[2])
            fh = int(face_info[3])
            x2 = min(w, x1 + fw)
            y2 = min(h, y1 + fh)
            score = float(face_info[14]) if len(face_info) > 14 else 0.9

            # Extract 128-D face embedding using SFace
            aligned = recognizer.alignCrop(img, face_info)
            embedding = recognizer.feature(aligned)
            embedding = embedding.flatten().astype(np.float32)

            faces.append({
                'bbox': [x1, y1, x2, y2],
                'embedding': embedding,
                'confidence': score
            })
    return faces

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

# ── Anti-Spoofing: Enhanced Multi-Layer Liveness Detection ──
LIVENESS_LAPLACIAN_THRESHOLD = 50  # Raised from 35 for stricter detection

def check_liveness(img, bbox):
    """Enhanced multi-layer anti-spoofing check.
    Combines 6 different analysis methods for robust fake detection.
    Returns (is_live, score, reason).
    """
    try:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        h, w = img.shape[:2]
        x2, y2 = min(x2, w), min(y2, h)
        face_roi = img[y1:y2, x1:x2]

        if face_roi.size == 0 or face_roi.shape[0] < 30 or face_roi.shape[1] < 30:
            return True, 100, 'too_small'

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        checks_passed = 0
        total_checks = 6

        # 1. Laplacian variance (texture sharpness) — threshold raised to 50
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        if lap_var > LIVENESS_LAPLACIAN_THRESHOLD:
            checks_passed += 1

        # 2. Edge micro-texture density
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size * 100
        if edge_ratio > 4:
            checks_passed += 1

        # 3. Color space analysis (YCrCb) — screens have different color distributions
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        cr_std = np.std(ycrcb[:, :, 1])
        cb_std = np.std(ycrcb[:, :, 2])
        if cr_std > 8 and cb_std > 8:
            checks_passed += 1

        # 4. Frequency domain analysis (DCT) — photos/screens lack high-freq content
        resized = cv2.resize(gray, (64, 64))
        dct = cv2.dct(np.float32(resized))
        high_freq_energy = np.sum(np.abs(dct[32:, 32:])) / np.sum(np.abs(dct) + 1e-6)
        if high_freq_energy > 0.01:
            checks_passed += 1

        # 5. Moiré pattern detection — screens produce periodic grid artifacts
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        center = magnitude[magnitude.shape[0]//4:3*magnitude.shape[0]//4,
                           magnitude.shape[1]//4:3*magnitude.shape[1]//4]
        peak_ratio = np.max(center) / (np.mean(center) + 1e-6)
        if peak_ratio < 15:  # Screens have sharper periodic peaks
            checks_passed += 1

        # 6. Reflection / glare detection — phones/screens have bright spots
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        bright_mask = (hsv[:, :, 2] > 240).astype(np.float32)
        glare_ratio = np.sum(bright_mask) / bright_mask.size
        if glare_ratio < 0.08:  # Real faces rarely have large glare patches
            checks_passed += 1

        # Combined score
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        score = (lap_var * 0.3) + (edge_ratio * 2.5) + (h_std * 0.3) + (s_std * 0.1) + (checks_passed * 10)

        is_live = checks_passed >= 4  # Must pass at least 4 of 6 checks
        reason = 'live' if is_live else f'photo_suspected({checks_passed}/6)'
        print(f"[LIVENESS] Checks: {checks_passed}/{total_checks} | lap={lap_var:.1f} edge={edge_ratio:.1f} cr={cr_std:.1f} cb={cb_std:.1f} hf={high_freq_energy:.4f} glare={glare_ratio:.3f}")
        return is_live, round(score, 1), reason

    except Exception as e:
        print(f"[LIVENESS] Error: {e}")
        return True, 100, 'error'

def check_face_uniqueness(embedding, exclude_uid=None, exclude_role=None):
    """Check if a face embedding is already registered in ANY database.
    Returns (is_unique, matched_name, matched_role) tuple.
    """
    UNIQUENESS_THRESHOLD = 0.55  # Higher than login threshold to avoid false positives

    role_db_map = {
        'student': ('students', 'student'),
        'teacher': ('teachers', 'teacher'),
        'admin': ('admins', 'admin')
    }

    emb_dim = len(embedding)

    for table, role in role_db_map.values():
        try:
            conn = get_db(role)
            rows = conn.execute(f'SELECT uid, name, face_embeddings FROM {table} WHERE face_registered = 1').fetchall()
            conn.close()

            for row in rows:
                if exclude_uid and row['uid'] == exclude_uid:
                    continue
                if row['face_embeddings']:
                    stored = np.frombuffer(row['face_embeddings'], dtype=np.float32)
                    num_embs = len(stored) // emb_dim
                    if num_embs > 0:
                        stored = stored.reshape(num_embs, emb_dim)
                        for emb in stored:
                            sim = cosine_similarity(embedding, emb)
                            if sim >= UNIQUENESS_THRESHOLD:
                                return False, row['name'], role
        except Exception as e:
            print(f"[UNIQUENESS] Error checking {role}: {e}")
            continue

    return True, None, None

# ── JWT Helpers ────────────────────────────────────────────
def create_token(uid, role, name):
    payload = {
        'uid': uid, 'role': role, 'name': name,
        'exp': datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm='HS256')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'success': False, 'message': 'Token missing'}), 401
        try:
            data = jwt.decode(token, Config.JWT_SECRET, algorithms=['HS256'])
            request.user = data
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'message': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'message': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ── Serve Frontend ─────────────────────────────────────────
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# ── Login Attempt Tracking ─────────────────────────────────
login_attempts = {}  # {uid: {'count': int, 'locked_until': datetime}}
LOCKOUT_MINUTES = 15
MAX_ATTEMPTS = 5

def check_login_lock(uid):
    """Check if account is locked. Returns (is_locked, minutes_remaining)."""
    record = login_attempts.get(uid)
    if not record:
        return False, 0
    if record.get('locked_until') and datetime.now() < record['locked_until']:
        remaining = (record['locked_until'] - datetime.now()).total_seconds() / 60
        return True, round(remaining, 1)
    if record.get('locked_until') and datetime.now() >= record['locked_until']:
        login_attempts.pop(uid, None)
    return False, 0

def record_failed_login(uid):
    """Record a failed login attempt. Lock after MAX_ATTEMPTS."""
    if uid not in login_attempts:
        login_attempts[uid] = {'count': 0}
    login_attempts[uid]['count'] += 1
    if login_attempts[uid]['count'] >= MAX_ATTEMPTS:
        login_attempts[uid]['locked_until'] = datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)
        return True
    return False

def clear_login_attempts(uid):
    login_attempts.pop(uid, None)

# ── AUTH: UID + Password Login ─────────────────────────────
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    uid = data.get('uid', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'student')

    if not uid or not password:
        return jsonify({'success': False, 'message': 'UID and password required'}), 400

    # Check if account is locked
    is_locked, mins_left = check_login_lock(uid)
    if is_locked:
        return jsonify({'success': False, 'message': f'Account locked. Try again in {mins_left} minutes.'}), 429

    table_map = {'student': 'students', 'teacher': 'teachers', 'admin': 'admins'}
    table = table_map.get(role)
    if not table:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400

    try:
        conn = get_db(role)
        user = conn.execute(f'SELECT * FROM {table} WHERE uid = ?', (uid,)).fetchone()
        conn.close()

        if not user:
            record_failed_login(uid)
            return jsonify({'success': False, 'message': 'User not found'}), 404

        if not check_password(password, user['password_hash']):
            was_locked = record_failed_login(uid)
            remaining = MAX_ATTEMPTS - login_attempts.get(uid, {}).get('count', 0)
            msg = 'Incorrect password'
            if was_locked:
                msg = f'Account locked for {LOCKOUT_MINUTES} minutes due to too many failed attempts.'
            elif remaining <= 2:
                msg = f'Incorrect password. {remaining} attempts remaining.'
            return jsonify({'success': False, 'message': msg}), 401

        # Success — clear attempts
        clear_login_attempts(uid)
        token = create_token(uid, role, user['name'])
        user_data = {'uid': uid, 'name': user['name'], 'role': role,
                     'face_registered': bool(user['face_registered'])}
        if role != 'admin':
            user_data['email'] = user['email'] or ''

        _audit(uid, role, 'login', 'Password login successful')
        return jsonify({'success': True, 'token': token, 'user': user_data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ── AUTH: Face Login ───────────────────────────────────────
@app.route('/api/auth/face-login', methods=['POST'])
def face_login():
    data = request.get_json()
    image_b64 = data.get('image', '')
    role = data.get('role', 'student')

    try:
        # Decode image
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image'}), 400

        faces = detect_faces_mediapipe(img)
        if not faces:
            return jsonify({'success': False, 'message': 'No face detected. Please look at the camera.'}), 400

        # Anti-spoofing: check liveness
        is_live, live_score, live_reason = check_liveness(img, faces[0]['bbox'])
        if not is_live:
            return jsonify({'success': False, 'message': 'Photo detected! Please use your real face, not a photo or screen.'}), 403

        query_embedding = faces[0]['embedding']
        if query_embedding is None:
            return jsonify({'success': False, 'message': 'Could not extract face features'}), 400

        # Search in database
        table_map = {'student': 'students', 'teacher': 'teachers', 'admin': 'admins'}
        table = table_map.get(role)
        conn = get_db(role)
        users = conn.execute(f'SELECT uid, name, face_embeddings FROM {table} WHERE face_registered = 1').fetchall()
        conn.close()

        best_match, best_sim = None, -1
        emb_dim = len(query_embedding)
        for user in users:
            if user['face_embeddings']:
                stored = np.frombuffer(user['face_embeddings'], dtype=np.float32)
                num_embs = len(stored) // emb_dim
                if num_embs > 0:
                    stored = stored.reshape(num_embs, emb_dim)
                    for emb in stored:
                        sim = cosine_similarity(query_embedding, emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_match = user

        if best_match and best_sim >= Config.FACE_SIMILARITY_THRESHOLD:
            token = create_token(best_match['uid'], role, best_match['name'])
            _audit(best_match['uid'], role, 'face_login', f'Face login (confidence: {round(float(best_sim), 3)})')
            return jsonify({'success': True, 'token': token,
                           'user': {'uid': best_match['uid'], 'name': best_match['name'],
                                    'role': role, 'confidence': round(float(best_sim), 3)}})
        else:
            return jsonify({'success': False, 'message': f'Face not recognized (best: {round(best_sim, 3)})'}), 401
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

# ── AUTH: Register User ───────────────────────────────────
@app.route('/api/auth/register', methods=['POST'])
@token_required
def register_user():
    caller = request.user
    if caller['role'] not in ('admin', 'teacher'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    data = request.get_json()
    role = data.get('role', 'student')
    uid = data.get('uid', '').strip()
    name = data.get('name', '').strip()
    password = data.get('password', '')

    if not uid or not name or not password:
        return jsonify({'success': False, 'message': 'UID, name, and password required'}), 400

    if caller['role'] == 'teacher' and role != 'student':
        return jsonify({'success': False, 'message': 'Teachers can only register students'}), 403

    table_map = {'student': 'students', 'teacher': 'teachers', 'admin': 'admins'}
    table = table_map.get(role)
    if not table:
        return jsonify({'success': False, 'message': 'Invalid role'}), 400

    try:
        conn = get_db(role)
        existing = conn.execute(f'SELECT uid FROM {table} WHERE uid = ?', (uid,)).fetchone()
        if existing:
            conn.close()
            return jsonify({'success': False, 'message': 'UID already exists'}), 409

        pw_hash = hash_password(password)

        if role == 'student':
            conn.execute('''INSERT INTO students (uid, name, email, phone, section, semester, department, password_hash)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                        (uid, name, data.get('email',''), data.get('phone',''),
                         data.get('section',''), data.get('semester',''), data.get('department',''), pw_hash))
        elif role == 'teacher':
            conn.execute('''INSERT INTO teachers (uid, name, email, phone, department, password_hash)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                        (uid, name, data.get('email',''), data.get('phone',''), data.get('department',''), pw_hash))
        elif role == 'admin':
            conn.execute('INSERT INTO admins (uid, name, email, password_hash) VALUES (?, ?, ?, ?)',
                        (uid, name, data.get('email',''), pw_hash))

        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'{role.title()} registered successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ── AUTH: Register Face ────────────────────────────────────
@app.route('/api/auth/face-register', methods=['POST'])
@token_required
def face_register():
    data = request.get_json()
    target_uid = data.get('uid', request.user['uid'])
    target_role = data.get('role', request.user['role'])
    images = data.get('images', [])

    if not images:
        return jsonify({'success': False, 'message': 'No images provided'}), 400

    try:
        embeddings = []
        for img_b64 in images:
            if ',' in img_b64:
                img_b64 = img_b64.split(',')[1]
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            faces = detect_faces_mediapipe(img)
            if faces and faces[0]['embedding'] is not None:
                # Anti-spoofing: check liveness on each capture
                is_live, live_score, live_reason = check_liveness(img, faces[0]['bbox'])
                if not is_live:
                    return jsonify({'success': False,
                                   'message': 'Photo/screen detected! Please use your real face for registration.'}), 403
                embeddings.append(faces[0]['embedding'])
                print(f"[FACE] Captured embedding {len(embeddings)} for {target_uid} (dim={len(faces[0]['embedding'])})")

        if not embeddings:
            return jsonify({'success': False, 'message': 'No faces detected in provided images. Please ensure your face is clearly visible.'}), 400

        # Face uniqueness check: ensure this face isn't already registered elsewhere
        is_unique, dup_name, dup_role = check_face_uniqueness(embeddings[0], exclude_uid=target_uid)
        if not is_unique:
            return jsonify({'success': False,
                           'message': f'This face is already registered as {dup_name} ({dup_role}). Each person must have a unique face.'}), 409

        emb_array = np.array(embeddings, dtype=np.float32)
        emb_blob = emb_array.tobytes()

        table_map = {'student': 'students', 'teacher': 'teachers', 'admin': 'admins'}
        table = table_map.get(target_role)
        conn = get_db(target_role)
        conn.execute(f'UPDATE {table} SET face_embeddings = ?, face_registered = 1 WHERE uid = ?',
                    (emb_blob, target_uid))
        conn.commit()
        conn.close()

        # Save first capture as profile photo
        try:
            profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'profiles')
            os.makedirs(profile_dir, exist_ok=True)
            first_img = images[0]
            if ',' in first_img:
                first_img = first_img.split(',')[1]
            photo_bytes = base64.b64decode(first_img)
            photo_path = os.path.join(profile_dir, f'{target_uid}.jpg')
            with open(photo_path, 'wb') as f:
                f.write(photo_bytes)
            print(f"[FACE] Profile photo saved for {target_uid}")
        except Exception as pe:
            print(f"[FACE] Profile photo save failed: {pe}")

        print(f"[FACE] [OK] Registered {len(embeddings)} embeddings for {target_uid}")
        return jsonify({'success': True, 'message': f'Face registered successfully! ({len(embeddings)} captures saved)'})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

# ── AUTH: Verify Token ─────────────────────────────────────
@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token():
    return jsonify({'success': True, 'user': request.user})

# ── PROFILE PHOTO ──────────────────────────────────────────
@app.route('/api/profile/photo/<uid>')
def get_profile_photo(uid):
    """Serve profile photo for a user."""
    profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'profiles')
    photo_path = os.path.join(profile_dir, f'{uid}.jpg')
    if os.path.exists(photo_path):
        return send_from_directory(profile_dir, f'{uid}.jpg', mimetype='image/jpeg')
    else:
        # Return a default SVG placeholder
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
            <rect width="200" height="200" fill="#1e293b"/>
            <text x="100" y="115" text-anchor="middle" fill="#64748b" font-size="72" font-family="sans-serif">{uid[0].upper() if uid else "?"}</text>
        </svg>'''
        from flask import Response
        return Response(svg, mimetype='image/svg+xml')

# ── TIMETABLE CRUD ─────────────────────────────────────────
@app.route('/api/timetable', methods=['GET'])
@token_required
def get_timetable():
    uid = request.user['uid']
    conn = get_db('teacher')
    rows = conn.execute('SELECT * FROM timetable WHERE teacher_uid = ? ORDER BY day_of_week, start_time', (uid,)).fetchall()
    conn.close()
    return jsonify({'success': True, 'timetable': [dict(r) for r in rows]})

@app.route('/api/timetable', methods=['POST'])
@token_required
def create_timetable():
    data = request.get_json()
    tid = str(uuid.uuid4())[:8]
    conn = get_db('teacher')
    conn.execute('''INSERT INTO timetable (id, teacher_uid, subject, section, day_of_week, start_time, end_time, duration_min, room)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (tid, request.user['uid'], data['subject'], data['section'],
                 data['day_of_week'], data['start_time'], data['end_time'],
                 data.get('duration_min', 50), data.get('room', '')))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'id': tid})

@app.route('/api/timetable/<tid>', methods=['DELETE'])
@token_required
def delete_timetable(tid):
    conn = get_db('teacher')
    conn.execute('DELETE FROM timetable WHERE id = ? AND teacher_uid = ?', (tid, request.user['uid']))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ── STUDENTS CRUD ──────────────────────────────────────────
@app.route('/api/students', methods=['GET'])
@token_required
def get_students():
    section = request.args.get('section', '')
    conn = get_db('student')
    if section:
        rows = conn.execute('SELECT uid, name, email, section, semester, department, face_registered FROM students WHERE section = ? AND is_active = 1', (section,)).fetchall()
    else:
        rows = conn.execute('SELECT uid, name, email, section, semester, department, face_registered FROM students WHERE is_active = 1').fetchall()
    conn.close()
    return jsonify({'success': True, 'students': [dict(r) for r in rows]})

@app.route('/api/students/<uid>', methods=['GET'])
@token_required
def get_student(uid):
    conn = get_db('student')
    s = conn.execute('SELECT uid, name, email, phone, section, semester, department, face_registered, created_at FROM students WHERE uid = ?', (uid,)).fetchone()
    conn.close()
    if not s:
        return jsonify({'success': False, 'message': 'Student not found'}), 404
    return jsonify({'success': True, 'student': dict(s)})

# ── ATTENDANCE SESSIONS ────────────────────────────────────
session_timers = {}  # {session_id: greenlet_timer}
MIN_PRESENCE_PERCENT = 75  # Default, can be overridden by settings

def auto_stop_session(session_id, duration_min, teacher_uid, subject, section):
    """Auto-stop a session after its duration expires. Calculates presence for all tracked students."""
    print(f"[SESSION] Auto-stopping session {session_id} after {duration_min} min")
    now = datetime.now()

    # Mark session as completed in DB
    conn = get_db('teacher')
    conn.execute("UPDATE attendance_sessions SET status = 'completed', actual_end = ? WHERE id = ?",
                (now.isoformat(), session_id))
    conn.commit()
    conn.close()

    # Calculate attendance from trackers
    finalize_session_attendance(session_id, duration_min, teacher_uid, subject, section)

    # Notify connected client
    try:
        for sid_key, sess_data in list(active_sessions.items()):
            if sess_data.get('session_id') == session_id:
                socketio.emit('session_auto_stopped', {
                    'session_id': session_id,
                    'message': f'Session auto-stopped after {duration_min} minutes'
                }, room=sid_key)
    except Exception as e:
        print(f"[SESSION] Error notifying client: {e}")

    session_timers.pop(session_id, None)

def finalize_session_attendance(session_id, duration_min, teacher_uid, subject, section):
    """Calculate and save attendance records for all students based on tracked presence."""
    duration_seconds = duration_min * 60
    if duration_seconds <= 0:
        return

    # Find the tracker data for this session
    trackers = {}
    # First check persistent store
    if session_id in session_tracker_store:
        trackers = session_tracker_store[session_id]
    else:
        # Fallback to active sessions
        for sid_key, sess_data in list(active_sessions.items()):
            if sess_data.get('session_id') == session_id:
                trackers = sess_data.get('trackers', {})
                break

    # Get all enrolled students in section
    sconn = get_db('student')
    enrolled = sconn.execute('SELECT uid, name FROM students WHERE section = ? AND is_active = 1',
                            (section,)).fetchall()

    total_present = 0
    total_absent = 0
    today = datetime.now().strftime('%Y-%m-%d')

    for student in enrolled:
        uid = student['uid']
        tracker = trackers.get(uid)

        if tracker:
            present_seconds = tracker.get('total_seconds', 0)
            presence_pct = round((present_seconds / duration_seconds) * 100, 1)
            present_min = round(present_seconds / 60, 2)
            status = 'present' if presence_pct >= MIN_PRESENCE_PERCENT else 'absent'
        else:
            present_seconds = 0
            presence_pct = 0
            present_min = 0
            status = 'absent'

        if status == 'present':
            total_present += 1
        else:
            total_absent += 1

        # Check if record already exists (from manual override)
        existing = sconn.execute('SELECT id FROM attendance WHERE student_uid = ? AND session_id = ?',
                                (uid, session_id)).fetchone()
        if existing:
            # Don't overwrite manual overrides
            continue

        sconn.execute('''INSERT INTO attendance (student_uid, session_id, subject, section, date,
                       class_duration_min, present_duration_min, presence_percentage, status, marked_by,
                       first_seen, last_seen)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (uid, session_id, subject, section, today,
                     duration_min, present_min, presence_pct, status, 'ai',
                     tracker['first_seen'].isoformat() if tracker else None,
                     tracker['last_seen'].isoformat() if tracker else None))

    sconn.commit()
    sconn.close()

    # Update session stats
    tconn = get_db('teacher')
    tconn.execute('UPDATE attendance_sessions SET total_enrolled = ?, total_present = ?, total_absent = ? WHERE id = ?',
                 (len(enrolled), total_present, total_absent, session_id))
    tconn.commit()
    tconn.close()
    print(f"[SESSION] Attendance saved: {total_present} present, {total_absent} absent out of {len(enrolled)}")
    # Clean up persistent tracker store
    session_tracker_store.pop(session_id, None)

@app.route('/api/attendance/start', methods=['POST'])
@token_required
def start_session():
    data = request.get_json()
    sid = str(uuid.uuid4())[:8]
    now = datetime.now()
    duration_min = int(data.get('duration_min', 50))
    subject = data.get('subject', '')
    section = data.get('section', '')
    scheduled_end = (now + timedelta(minutes=duration_min)).isoformat()

    conn = get_db('teacher')
    conn.execute('''INSERT INTO attendance_sessions
                   (id, timetable_id, teacher_uid, subject, section, date, actual_start,
                    scheduled_start, duration_min, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')''',
                (sid, data.get('timetable_id',''), request.user['uid'],
                 subject, section, now.strftime('%Y-%m-%d'), now.isoformat(),
                 scheduled_end, duration_min))
    conn.commit()
    conn.close()

    # Schedule auto-stop timer
    timer = eventlet.spawn_after(duration_min * 60, auto_stop_session,
                                  sid, duration_min, request.user['uid'], subject, section)
    session_timers[sid] = timer
    print(f"[SESSION] Started {sid} | {subject} | {section} | {duration_min}min | auto-stop at {scheduled_end}")

    # Broadcast session start alert to all connected clients
    try:
        socketio.emit('session_alert', {
            'type': 'session_started',
            'session_id': sid,
            'subject': subject,
            'section': section,
            'duration_min': duration_min,
            'teacher': request.user['name'],
            'message': f'📸 Attendance started: {subject} ({section}) — {duration_min} min'
        }, broadcast=True)
    except: pass

    return jsonify({'success': True, 'session_id': sid, 'duration_min': duration_min,
                   'scheduled_end': scheduled_end})

@app.route('/api/attendance/stop', methods=['POST'])
@token_required
def stop_session():
    data = request.get_json()
    sid = data.get('session_id')
    now = datetime.now().isoformat()

    # Cancel auto-stop timer
    timer = session_timers.pop(sid, None)
    if timer:
        timer.cancel()

    # Get session info for finalization
    conn = get_db('teacher')
    session_row = conn.execute('SELECT * FROM attendance_sessions WHERE id = ?', (sid,)).fetchone()
    conn.execute("UPDATE attendance_sessions SET status = 'completed', actual_end = ? WHERE id = ?", (now, sid))
    conn.commit()
    conn.close()

    # Finalize attendance with 75% rule
    if session_row:
        finalize_session_attendance(sid, session_row['duration_min'],
                                    session_row['teacher_uid'],
                                    session_row['subject'],
                                    session_row['section'])

    # Broadcast session stop
    try:
        socketio.emit('session_alert', {
            'type': 'session_stopped',
            'session_id': sid,
            'message': f'✅ Attendance session completed and records saved'
        }, broadcast=True)
    except: pass

    return jsonify({'success': True, 'message': 'Session stopped and attendance calculated'})

@app.route('/api/attendance/sessions', methods=['GET'])
@token_required
def get_sessions():
    conn = get_db('teacher')
    rows = conn.execute('SELECT * FROM attendance_sessions WHERE teacher_uid = ? ORDER BY date DESC LIMIT 50',
                       (request.user['uid'],)).fetchall()
    conn.close()
    return jsonify({'success': True, 'sessions': [dict(r) for r in rows]})

@app.route('/api/attendance/student/<uid>', methods=['GET'])
@token_required
def get_student_attendance(uid):
    conn = get_db('student')
    rows = conn.execute('SELECT * FROM attendance WHERE student_uid = ? ORDER BY date DESC LIMIT 100', (uid,)).fetchall()
    conn.close()
    return jsonify({'success': True, 'records': [dict(r) for r in rows]})

@app.route('/api/attendance/mark', methods=['POST'])
@token_required
def mark_attendance():
    data = request.get_json()
    conn = get_db('student')
    conn.execute('''INSERT INTO attendance (student_uid, session_id, subject, section, date, class_duration_min,
                   present_duration_min, presence_percentage, status, marked_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (data['student_uid'], data['session_id'], data.get('subject',''), data.get('section',''),
                 data.get('date', datetime.now().strftime('%Y-%m-%d')),
                 data.get('class_duration_min', 50), data.get('present_duration_min', 0),
                 data.get('presence_percentage', 0), data.get('status', 'present'),
                 request.user['uid']))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ── MANUAL ATTENDANCE OVERRIDE ─────────────────────────────
@app.route('/api/attendance/override', methods=['POST'])
@token_required
def attendance_override():
    """Manual override: teacher can force-mark a student present or absent."""
    if request.user['role'] not in ('teacher', 'admin'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    data = request.get_json()
    student_uid = data.get('student_uid')
    session_id = data.get('session_id')
    new_status = data.get('status', 'manual_present')  # 'manual_present' or 'manual_absent'

    if not student_uid or not session_id:
        return jsonify({'success': False, 'message': 'student_uid and session_id required'}), 400

    conn = get_db('student')
    # Check if record exists
    existing = conn.execute('SELECT id FROM attendance WHERE student_uid = ? AND session_id = ?',
                           (student_uid, session_id)).fetchone()

    if existing:
        # Update existing record
        conn.execute('''UPDATE attendance SET status = ?, marked_by = ?, presence_percentage = ?
                       WHERE student_uid = ? AND session_id = ?''',
                    (new_status, request.user['uid'],
                     100 if 'present' in new_status else 0,
                     student_uid, session_id))
    else:
        # Get session info
        tconn = get_db('teacher')
        session_row = tconn.execute('SELECT * FROM attendance_sessions WHERE id = ?', (session_id,)).fetchone()
        tconn.close()

        conn.execute('''INSERT INTO attendance (student_uid, session_id, subject, section, date,
                       class_duration_min, present_duration_min, presence_percentage, status, marked_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (student_uid, session_id,
                     session_row['subject'] if session_row else '',
                     session_row['section'] if session_row else '',
                     datetime.now().strftime('%Y-%m-%d'),
                     session_row['duration_min'] if session_row else 50,
                     session_row['duration_min'] if session_row and 'present' in new_status else 0,
                     100 if 'present' in new_status else 0,
                     new_status, request.user['uid']))

    conn.commit()
    conn.close()

    _audit(request.user['uid'], request.user['role'], 'attendance_override',
           f'{new_status} for {student_uid} in session {session_id}')

    return jsonify({'success': True, 'message': f'Student {student_uid} marked as {new_status}'})

# ── ASSIGNMENTS ────────────────────────────────────────────
@app.route('/api/assignments', methods=['GET'])
@token_required
def get_assignments():
    role = request.user['role']
    if role == 'teacher':
        conn = get_db('teacher')
        rows = conn.execute('SELECT * FROM assignments WHERE teacher_uid = ? ORDER BY created_at DESC',
                           (request.user['uid'],)).fetchall()
    else:
        conn = get_db('student')
        section = request.args.get('section', '')
        if section:
            rows = conn.execute('SELECT * FROM student_assignments WHERE section = ? ORDER BY created_at DESC', (section,)).fetchall()
        else:
            rows = conn.execute('SELECT * FROM student_assignments ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify({'success': True, 'assignments': [dict(r) for r in rows]})

@app.route('/api/assignments', methods=['POST'])
@token_required
def create_assignment():
    data = request.get_json()
    aid = str(uuid.uuid4())[:8]
    conn = get_db('teacher')
    conn.execute('''INSERT INTO assignments (id, teacher_uid, title, description, subject, section, due_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (aid, request.user['uid'], data['title'], data.get('description',''),
                 data.get('subject',''), data.get('section',''), data.get('due_date','')))
    conn.commit()
    conn.close()

    # Also insert into student assignments
    sconn = get_db('student')
    sconn.execute('''INSERT INTO student_assignments (assignment_id, title, description, subject, section, due_date, uploaded_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                 (aid, data['title'], data.get('description',''), data.get('subject',''),
                  data.get('section',''), data.get('due_date',''), request.user['uid']))
    sconn.commit()
    sconn.close()
    return jsonify({'success': True, 'id': aid})

# ── ASSIGNMENT FILE UPLOAD ─────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, 'assignments'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, 'submissions'), exist_ok=True)

@app.route('/api/assignments/upload', methods=['POST'])
@token_required
def upload_assignment_file():
    """Teacher uploads an assignment file."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    aid = request.form.get('assignment_id', str(uuid.uuid4())[:8])
    title = request.form.get('title', file.filename)
    description = request.form.get('description', '')
    subject = request.form.get('subject', '')
    section = request.form.get('section', '')
    due_date = request.form.get('due_date', '')

    # Save file
    safe_name = f"{aid}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, 'assignments', safe_name)
    file.save(filepath)

    # Save to teacher DB
    conn = get_db('teacher')
    conn.execute('''INSERT OR REPLACE INTO assignments (id, teacher_uid, title, description, subject, section, due_date, file_path, file_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (aid, request.user['uid'], title, description, subject, section, due_date, filepath, file.filename))
    conn.commit()
    conn.close()

    # Save to student assignments DB
    sconn = get_db('student')
    sconn.execute('''INSERT OR REPLACE INTO student_assignments (assignment_id, title, description, subject, section, due_date, uploaded_by, file_path, file_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (aid, title, description, subject, section, due_date, request.user['uid'], filepath, file.filename))
    sconn.commit()
    sconn.close()

    return jsonify({'success': True, 'id': aid, 'message': 'Assignment uploaded successfully'})

@app.route('/api/assignments/<aid>/grade', methods=['PUT'])
@token_required
def grade_submission(aid):
    """Teacher grades a student submission."""
    data = request.get_json()
    student_uid = data.get('student_uid')
    grade = data.get('grade', '')
    feedback = data.get('feedback', '')

    conn = get_db('student')
    conn.execute('UPDATE assignment_submissions SET grade = ?, feedback = ? WHERE assignment_id = ? AND student_uid = ?',
                (grade, feedback, aid, student_uid))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Graded successfully'})

@app.route('/api/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_DIR, filename)


# ── ADMIN ROUTES ───────────────────────────────────────────
@app.route('/api/admin/stats', methods=['GET'])
@token_required
def admin_stats():
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403

    sconn = get_db('student')
    total_students = sconn.execute('SELECT COUNT(*) FROM students WHERE is_active = 1').fetchone()[0]
    sconn.close()

    tconn = get_db('teacher')
    total_teachers = tconn.execute('SELECT COUNT(*) FROM teachers WHERE is_active = 1').fetchone()[0]
    total_sessions = tconn.execute('SELECT COUNT(*) FROM attendance_sessions').fetchone()[0]
    active_sessions = tconn.execute("SELECT COUNT(*) FROM attendance_sessions WHERE status = 'active'").fetchone()[0]
    tconn.close()

    aconn = get_db('admin')
    total_sections = aconn.execute('SELECT COUNT(*) FROM sections').fetchone()[0]
    aconn.close()

    return jsonify({'success': True, 'stats': {
        'total_students': total_students, 'total_teachers': total_teachers,
        'total_sessions': total_sessions, 'active_sessions': active_sessions,
        'total_sections': total_sections
    }})

@app.route('/api/admin/users', methods=['GET'])
@token_required
def admin_get_users():
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403

    role_filter = request.args.get('role', 'student')
    table_map = {'student': 'students', 'teacher': 'teachers'}
    table = table_map.get(role_filter, 'students')

    conn = get_db(role_filter if role_filter in ('student','teacher') else 'student')
    if role_filter == 'student':
        rows = conn.execute('SELECT uid, name, email, section, semester, department, face_registered, is_active, created_at FROM students').fetchall()
    else:
        rows = conn.execute('SELECT uid, name, email, department, face_registered, is_active, created_at FROM teachers').fetchall()
    conn.close()
    return jsonify({'success': True, 'users': [dict(r) for r in rows]})

@app.route('/api/admin/sections', methods=['GET'])
@token_required
def get_sections():
    conn = get_db('admin')
    rows = conn.execute('SELECT * FROM sections ORDER BY name').fetchall()
    conn.close()
    return jsonify({'success': True, 'sections': [dict(r) for r in rows]})

@app.route('/api/admin/sections', methods=['POST'])
@token_required
def create_section():
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    data = request.get_json()
    sid = str(uuid.uuid4())[:8]
    conn = get_db('admin')
    conn.execute('INSERT INTO sections (id, name, department, semester, academic_year) VALUES (?, ?, ?, ?, ?)',
                (sid, data['name'], data.get('department',''), data.get('semester',''), data.get('academic_year','')))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'id': sid})

# ── ATTENDANCE REPORTS ─────────────────────────────────────
@app.route('/api/attendance/report', methods=['GET'])
@token_required
def attendance_report():
    section = request.args.get('section', '')
    subject = request.args.get('subject', '')
    date_from = request.args.get('from', '')
    date_to = request.args.get('to', '')

    conn = get_db('student')
    query = 'SELECT * FROM attendance WHERE 1=1'
    params = []
    if section:
        query += ' AND section = ?'; params.append(section)
    if subject:
        query += ' AND subject = ?'; params.append(subject)
    if date_from:
        query += ' AND date >= ?'; params.append(date_from)
    if date_to:
        query += ' AND date <= ?'; params.append(date_to)
    query += ' ORDER BY date DESC, student_uid'

    rows = conn.execute(query, params).fetchall()
    conn.close()

    records = [dict(r) for r in rows]

    # Compute summary
    students_set = set(r['student_uid'] for r in records)
    present = sum(1 for r in records if r['status'] == 'present')
    absent = sum(1 for r in records if r['status'] == 'absent')

    return jsonify({'success': True, 'records': records,
                    'summary': {'total_records': len(records), 'unique_students': len(students_set),
                                'present': present, 'absent': absent,
                                'avg_percentage': round(sum(r.get('presence_percentage',0) or 0 for r in records) / max(len(records),1), 1)}})

@app.route('/api/attendance/report/csv', methods=['GET'])
@token_required
def attendance_csv():
    section = request.args.get('section', '')
    subject = request.args.get('subject', '')
    date_from = request.args.get('from', '')
    date_to = request.args.get('to', '')

    conn = get_db('student')
    query = 'SELECT student_uid, subject, section, date, class_duration_min, present_duration_min, presence_percentage, status, marked_by FROM attendance WHERE 1=1'
    params = []
    if section: query += ' AND section = ?'; params.append(section)
    if subject: query += ' AND subject = ?'; params.append(subject)
    if date_from: query += ' AND date >= ?'; params.append(date_from)
    if date_to: query += ' AND date <= ?'; params.append(date_to)
    query += ' ORDER BY date DESC'

    rows = conn.execute(query, params).fetchall()
    conn.close()

    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Student UID', 'Subject', 'Section', 'Date', 'Class Duration (min)', 'Present Duration (min)', 'Presence %', 'Status', 'Marked By'])
    for r in rows:
        writer.writerow([r['student_uid'], r['subject'], r['section'], r['date'],
                        r['class_duration_min'], r['present_duration_min'],
                        r['presence_percentage'], r['status'], r['marked_by']])

    from flask import Response
    return Response(output.getvalue(), mimetype='text/csv',
                   headers={'Content-Disposition': f'attachment; filename=attendance_report_{datetime.now().strftime("%Y%m%d")}.csv'})

# ── MANUAL ATTENDANCE OVERRIDE ─────────────────────────────
@app.route('/api/attendance/manual', methods=['POST'])
@token_required
def manual_attendance():
    if request.user['role'] not in ('teacher', 'admin'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    data = request.get_json()
    conn = get_db('student')
    conn.execute('''INSERT INTO attendance (student_uid, session_id, subject, section, date,
                   class_duration_min, present_duration_min, presence_percentage, status, marked_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (data['student_uid'], data.get('session_id','manual'), data.get('subject',''),
                 data.get('section',''), data.get('date', datetime.now().strftime('%Y-%m-%d')),
                 data.get('class_duration_min',50), data.get('present_duration_min',50),
                 100 if data.get('status','present')=='present' else 0,
                 data.get('status','present'), request.user['uid']))
    conn.commit()
    conn.close()
    _audit(request.user['uid'], request.user['role'], 'manual_attendance',
           f"Marked {data['student_uid']} as {data.get('status','present')}")
    return jsonify({'success': True})

# ── AUDIT LOG ──────────────────────────────────────────────
def _audit(uid, role, action, details=''):
    try:
        ip = ''
        try:
            ip = request.remote_addr or request.environ.get('HTTP_X_FORWARDED_FOR', '')
        except: pass
        conn = get_db('admin')
        conn.execute('INSERT INTO audit_log (user_uid, user_role, action, details, ip_address) VALUES (?, ?, ?, ?, ?)',
                    (uid, role, action, details, ip))
        conn.commit()
        conn.close()
    except: pass

@app.route('/api/admin/audit', methods=['GET'])
@token_required
def get_audit_log():
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    limit = request.args.get('limit', 100, type=int)
    conn = get_db('admin')
    rows = conn.execute('SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return jsonify({'success': True, 'logs': [dict(r) for r in rows]})

# ── DATABASE BACKUP & RESTORE ─────────────────────────────
@app.route('/api/admin/backup', methods=['GET'])
@token_required
def backup_database():
    """Download a zip of all databases."""
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403

    import zipfile, io
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for db_name in ['students.db', 'teachers.db', 'admin.db']:
            db_path = os.path.join(Config.DATABASE_DIR, db_name)
            if os.path.exists(db_path):
                zf.write(db_path, db_name)

    zip_buffer.seek(0)
    from flask import Response
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Response(zip_buffer.getvalue(),
                   mimetype='application/zip',
                   headers={'Content-Disposition': f'attachment; filename=attendai_backup_{timestamp}.zip'})

@app.route('/api/admin/restore', methods=['POST'])
@token_required
def restore_database():
    """Restore databases from a zip backup."""
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No backup file provided'}), 400

    import zipfile, io
    file = request.files['file']
    zip_buffer = io.BytesIO(file.read())

    try:
        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            valid_names = {'students.db', 'teachers.db', 'admin.db'}
            for name in zf.namelist():
                if name in valid_names:
                    db_path = os.path.join(Config.DATABASE_DIR, name)
                    with open(db_path, 'wb') as f:
                        f.write(zf.read(name))
        _audit(request.user['uid'], 'admin', 'database_restore', 'Restored from backup')
        return jsonify({'success': True, 'message': 'Database restored successfully. Please restart the server.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Invalid backup file: {str(e)}'}), 400

# ── BULK CSV STUDENT IMPORT ────────────────────────────────
@app.route('/api/admin/import-students', methods=['POST'])
@token_required
def import_students_csv():
    """Import students from a CSV file. Expected headers: uid, name, email, phone, section, semester, department, password"""
    if request.user['role'] not in ('admin', 'teacher'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No CSV file provided'}), 400

    import csv, io
    file = request.files['file']
    content = file.read().decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))

    conn = get_db('student')
    imported = 0
    skipped = 0
    errors = []

    for row in reader:
        uid = row.get('uid', '').strip()
        name = row.get('name', '').strip()
        password = row.get('password', '').strip()

        if not uid or not name:
            errors.append(f"Row missing uid/name: {row}")
            skipped += 1
            continue

        existing = conn.execute('SELECT uid FROM students WHERE uid = ?', (uid,)).fetchone()
        if existing:
            skipped += 1
            continue

        pw_hash = hash_password(password or 'student123')  # Default password
        conn.execute('''INSERT INTO students (uid, name, email, phone, section, semester, department, password_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (uid, name, row.get('email',''), row.get('phone',''),
                     row.get('section',''), row.get('semester',''),
                     row.get('department',''), pw_hash))
        imported += 1

    conn.commit()
    conn.close()
    _audit(request.user['uid'], request.user['role'], 'csv_import', f'Imported {imported} students, {skipped} skipped')

    return jsonify({'success': True, 'imported': imported, 'skipped': skipped,
                   'errors': errors[:10], 'message': f'{imported} students imported, {skipped} skipped'})

# ── ADMIN: Toggle/Delete User ──────────────────────────────
@app.route('/api/admin/users/<uid>/toggle', methods=['POST'])
@token_required
def toggle_user(uid):
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    data = request.get_json()
    role = data.get('role', 'student')
    table_map = {'student': 'students', 'teacher': 'teachers'}
    table = table_map.get(role, 'students')
    conn = get_db(role if role in ('student','teacher') else 'student')
    conn.execute(f'UPDATE {table} SET is_active = CASE WHEN is_active = 1 THEN 0 ELSE 1 END WHERE uid = ?', (uid,))
    conn.commit()
    conn.close()
    _audit(request.user['uid'], 'admin', 'toggle_user', f'Toggled {role} {uid}')
    return jsonify({'success': True})

@app.route('/api/admin/users/<uid>', methods=['DELETE'])
@token_required
def delete_user(uid):
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    role = request.args.get('role', 'student')
    # Also check body if provided
    body = request.get_json(silent=True, force=True) or {}
    if body.get('role'):
        role = body['role']
    table_map = {'student': 'students', 'teacher': 'teachers'}
    table = table_map.get(role, 'students')
    db_key = role if role in ('student', 'teacher') else 'student'
    conn = get_db(db_key)
    # Check user exists
    user_row = conn.execute(f'SELECT uid FROM {table} WHERE uid = ?', (uid,)).fetchone()
    if not user_row:
        conn.close()
        return jsonify({'success': False, 'message': f'User {uid} not found'}), 404
    conn.execute(f'DELETE FROM {table} WHERE uid = ?', (uid,))
    conn.commit()
    conn.close()
    # Also delete profile photo if exists
    try:
        photo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads', 'profiles', f'{uid}.jpg')
        if os.path.exists(photo_path):
            os.remove(photo_path)
    except: pass
    _audit(request.user['uid'], 'admin', 'delete_user', f'Deleted {role} {uid}')
    return jsonify({'success': True, 'message': f'{role.title()} {uid} deleted successfully'})

# ── STUDENT: Change Password ──────────────────────────────
@app.route('/api/auth/change-password', methods=['POST'])
@token_required
def change_password():
    data = request.get_json()
    old_pw = data.get('old_password', '')
    new_pw = data.get('new_password', '')
    if not old_pw or not new_pw or len(new_pw) < 6:
        return jsonify({'success': False, 'message': 'Valid passwords required (min 6 chars)'}), 400

    role = request.user['role']
    table_map = {'student': 'students', 'teacher': 'teachers', 'admin': 'admins'}
    table = table_map[role]
    conn = get_db(role)
    user = conn.execute(f'SELECT password_hash FROM {table} WHERE uid = ?', (request.user['uid'],)).fetchone()
    if not user or not check_password(old_pw, user['password_hash']):
        conn.close()
        return jsonify({'success': False, 'message': 'Current password is incorrect'}), 401
    new_hash = hash_password(new_pw)
    conn.execute(f'UPDATE {table} SET password_hash = ? WHERE uid = ?', (new_hash, request.user['uid']))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Password changed successfully'})

# ── ADMIN: Settings CRUD ──────────────────────────────────
@app.route('/api/admin/settings', methods=['GET'])
@token_required
def get_settings():
    conn = get_db('admin')
    rows = conn.execute('SELECT key, value FROM settings').fetchall()
    conn.close()
    return jsonify({'success': True, 'settings': {r['key']: r['value'] for r in rows}})

@app.route('/api/admin/settings', methods=['POST'])
@token_required
def update_settings():
    if request.user['role'] != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    data = request.get_json()
    conn = get_db('admin')
    for key, value in data.items():
        conn.execute('INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)',
                    (key, str(value), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    _audit(request.user['uid'], 'admin', 'update_settings', json.dumps(data))
    return jsonify({'success': True})

# ── SOCKET.IO: Live Attendance Stream ─────────────────────
active_sessions = {}          # {ws_sid: {session_id, trackers, start_time}}
session_tracker_store = {}    # {session_id: trackers} — persists after client disconnect

@socketio.on('connect')
def on_connect():
    print(f'[WS] Client connected: {request.sid}')

@socketio.on('start_stream')
def on_start_stream(data):
    session_id = data.get('session_id')
    trackers = {}
    active_sessions[request.sid] = {
        'session_id': session_id,
        'trackers': trackers,
        'start_time': datetime.now()
    }
    # Also store in persistent tracker store (survives disconnect)
    session_tracker_store[session_id] = trackers
    emit('stream_started', {'session_id': session_id})

@socketio.on('video_frame')
def on_video_frame(data):
    session = active_sessions.get(request.sid)
    if not session:
        return


    try:
        img_b64 = data.get('frame', '')
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return

        faces = detect_faces_mediapipe(img)
        results = []

        # Load students once per frame
        section = data.get('section', '')
        conn = get_db('student')
        if section:
            students = conn.execute('SELECT uid, name, face_embeddings FROM students WHERE face_registered = 1 AND section = ?', (section,)).fetchall()
        else:
            students = conn.execute('SELECT uid, name, face_embeddings FROM students WHERE face_registered = 1').fetchall()
        conn.close()

        for face in faces:
            bbox = face['bbox']
            embedding = face['embedding']
            matched_name = 'Unknown'
            matched_uid = ''
            best_sim = 0

            if embedding is not None:
                emb_dim = len(embedding)
                for s in students:
                    if s['face_embeddings']:
                        stored = np.frombuffer(s['face_embeddings'], dtype=np.float32)
                        num_embs = len(stored) // emb_dim
                        if num_embs > 0:
                            stored = stored.reshape(num_embs, emb_dim)
                            for emb in stored:
                                sim = cosine_similarity(embedding, emb)
                                if sim > best_sim:
                                    best_sim = sim
                                    if sim >= Config.FACE_SIMILARITY_THRESHOLD:
                                        matched_name = s['name']
                                        matched_uid = s['uid']

            # Update tracker (both in active_sessions AND session_tracker_store)
            if matched_uid:
                trackers = session['trackers']
                if matched_uid not in trackers:
                    trackers[matched_uid] = {'name': matched_name, 'first_seen': datetime.now(), 'last_seen': datetime.now(), 'total_seconds': 0}
                else:
                    prev = trackers[matched_uid]['last_seen']
                    now = datetime.now()
                    delta = (now - prev).total_seconds()
                    if delta < 60:
                        trackers[matched_uid]['total_seconds'] += delta
                    trackers[matched_uid]['last_seen'] = now

            # ── AI ATTENTIVENESS ANALYSIS ──
            attention = 50  # default
            mood = 'neutral'
            if matched_uid and embedding is not None:
                x, y, w, h = bbox
                img_h, img_w = img.shape[:2]
                
                # 1. Face size ratio (larger = closer/more attentive)
                face_area = (w * h) / (img_w * img_h)
                size_score = min(100, int(face_area * 800))  # normalize
                
                # 2. Face centering (centered = looking at screen/board)
                cx = (x + w/2) / img_w
                cy = (y + h/2) / img_h
                center_dist = ((cx - 0.5)**2 + (cy - 0.5)**2) ** 0.5
                center_score = max(0, int(100 - center_dist * 200))
                
                # 3. Face region sharpness (sharp = focused, blurry = moving)
                face_crop = img[max(0,y):min(img_h,y+h), max(0,x):min(img_w,x+w)]
                if face_crop.size > 0:
                    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
                    sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                    sharp_score = min(100, int(sharpness * 1.5))
                else:
                    sharp_score = 50
                
                # 4. Face aspect ratio (normal ~1.2-1.4, extreme = looking away)
                aspect = h / max(w, 1)
                aspect_score = 100 if 1.1 <= aspect <= 1.6 else max(0, int(100 - abs(aspect - 1.35) * 150))
                
                # 5. Brightness analysis (very dark = looking down, very bright = screen glare)
                if face_crop.size > 0:
                    brightness = np.mean(gray_face)
                    bright_score = 100 if 60 <= brightness <= 200 else max(0, int(100 - abs(brightness - 130) * 0.8))
                else:
                    bright_score = 50
                
                # Composite attention score
                attention = int(size_score * 0.15 + center_score * 0.25 + sharp_score * 0.25 + aspect_score * 0.2 + bright_score * 0.15)
                attention = max(10, min(100, attention))
                
                # Mood estimation from face metrics
                if attention >= 75:
                    mood = 'focused' if sharp_score > 60 else 'happy'
                elif attention >= 50:
                    mood = 'neutral' if center_score > 40 else 'confused'
                elif attention >= 30:
                    mood = 'bored' if sharp_score < 40 else 'stressed'
                else:
                    mood = 'bored'
                
                # Auto-save engagement to DB (throttled — every 5 seconds)
                tracker = session['trackers'].get(matched_uid, {})
                last_eng_update = tracker.get('last_eng_update', 0)
                now_ts = time.time()
                if now_ts - last_eng_update > 5:
                    try:
                        sess_id = session.get('session_id', '')
                        sconn = get_db('student')
                        existing = sconn.execute('SELECT attention_score, engagement_score, hand_raises, answers_given FROM student_engagement WHERE student_uid = ? AND session_id = ?',
                                               (matched_uid, sess_id)).fetchone()
                        if existing:
                            # Smooth with rolling average
                            old_att = existing['attention_score'] or 50
                            new_att = int(old_att * 0.6 + attention * 0.4)
                            eng = int(new_att * 0.7 + (existing['hand_raises'] or 0) * 5 + (existing['answers_given'] or 0) * 8)
                            eng = min(100, eng)
                            sconn.execute('UPDATE student_engagement SET attention_score = ?, mood = ?, engagement_score = ?, updated_at = CURRENT_TIMESTAMP WHERE student_uid = ? AND session_id = ?',
                                        (new_att, mood, eng, matched_uid, sess_id))
                        else:
                            eng = int(attention * 0.7)
                            sconn.execute('INSERT INTO student_engagement (student_uid, session_id, attention_score, mood, engagement_score) VALUES (?, ?, ?, ?, ?)',
                                        (matched_uid, sess_id, attention, mood, eng))
                        sconn.commit()
                        sconn.close()
                        tracker['last_eng_update'] = now_ts
                    except Exception as eng_err:
                        print(f'[ENGAGE] Error: {eng_err}')

            results.append({
                'bbox': bbox, 'name': matched_name, 'uid': matched_uid,
                'confidence': round(best_sim, 3),
                'attention': attention, 'mood': mood
            })

        emit('processed_frame', {'faces': results})
    except Exception as e:
        emit('processed_frame', {'faces': [], 'error': str(e)})

@socketio.on('stop_stream')
def on_stop_stream(data):
    session = active_sessions.pop(request.sid, None)
    if session:
        # Keep tracker data in persistent store for auto_stop/finalize
        emit('stream_stopped', {
            'trackers': {uid: {'name': t['name'], 'total_seconds': t['total_seconds']}
                        for uid, t in session['trackers'].items()}
        })

@socketio.on('disconnect')
def on_disconnect():
    # Don't delete tracker data — auto_stop_session may still need it
    active_sessions.pop(request.sid, None)
    print(f'[WS] Client disconnected: {request.sid}')

# ── ENHANCED ASSIGNMENT: Create with Files ─────────────────
@app.route('/api/assignments/create-with-files', methods=['POST'])
@token_required
def create_assignment_with_files():
    """Create assignment with multiple file attachments."""
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '')
    subject = request.form.get('subject', '')
    section = request.form.get('section', '')
    due_date = request.form.get('due_date', '')
    max_score = float(request.form.get('max_score', 100))

    if not title:
        return jsonify({'success': False, 'message': 'Title is required'}), 400

    aid = str(uuid.uuid4())[:8]
    files = request.files.getlist('files')
    attachment_count = len(files)

    # Save attachments
    saved_files = []
    for f in files:
        if f.filename:
            safe_name = f"{aid}_{f.filename}"
            fpath = os.path.join(UPLOAD_DIR, 'assignments', safe_name)
            f.save(fpath)
            saved_files.append({'name': f.filename, 'path': fpath, 'size': os.path.getsize(fpath)})

    # Save to teacher DB
    conn = get_db('teacher')
    conn.execute('''INSERT INTO assignments (id, teacher_uid, title, description, subject, section, due_date, max_score, attachment_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (aid, request.user['uid'], title, description, subject, section, due_date, max_score, attachment_count))
    for sf in saved_files:
        conn.execute('INSERT INTO assignment_attachments (assignment_id, file_path, file_name, file_size) VALUES (?, ?, ?, ?)',
                    (aid, sf['path'], sf['name'], sf['size']))
    conn.commit()
    conn.close()

    # Save to student DB
    sconn = get_db('student')
    sconn.execute('''INSERT OR REPLACE INTO student_assignments (assignment_id, title, description, subject, section, due_date, uploaded_by, max_score, attachment_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (aid, title, description, subject, section, due_date, request.user['uid'], max_score, attachment_count))
    for sf in saved_files:
        sconn.execute('INSERT INTO assignment_attachments (assignment_id, file_path, file_name, file_size) VALUES (?, ?, ?, ?)',
                     (aid, sf['path'], sf['name'], sf['size']))
    sconn.commit()
    sconn.close()

    return jsonify({'success': True, 'id': aid, 'message': f'Assignment created with {attachment_count} file(s)'})

@app.route('/api/assignments/<aid>/attachments', methods=['GET'])
@token_required
def get_assignment_attachments(aid):
    """Get all attachments for an assignment."""
    role = request.user['role']
    conn = get_db('student' if role == 'student' else 'teacher')
    rows = conn.execute('SELECT id, assignment_id, file_name, file_size, uploaded_at FROM assignment_attachments WHERE assignment_id = ?', (aid,)).fetchall()
    conn.close()
    return jsonify({'success': True, 'attachments': [dict(r) for r in rows]})

@app.route('/api/assignments/<aid>/download/<filename>')
def download_attachment(aid, filename):
    """Download an assignment attachment."""
    fpath = os.path.join(UPLOAD_DIR, 'assignments', f"{aid}_{filename}")
    if os.path.exists(fpath):
        directory = os.path.join(UPLOAD_DIR, 'assignments')
        return send_from_directory(directory, f"{aid}_{filename}", as_attachment=True, download_name=filename)
    return jsonify({'success': False, 'message': 'File not found'}), 404

@app.route('/api/assignments/<aid>/submit', methods=['POST'])
@token_required
def submit_assignment_enhanced(aid):
    """Student submits work — enhanced with multiple files."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    student_uid = request.user['uid']
    safe_name = f"{aid}_{student_uid}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, 'submissions', safe_name)
    file.save(filepath)

    conn = get_db('student')
    conn.execute('''INSERT OR REPLACE INTO assignment_submissions (assignment_id, student_uid, file_path, file_name)
                   VALUES (?, ?, ?, ?)''', (aid, student_uid, filepath, file.filename))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Submitted successfully'})

# ── AI ASSIGNMENT ANALYSIS (Groq) ─────────────────────────
def extract_text_from_file(filepath):
    """Extract text from PDF, DOCX, or return filename for images."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf' and PyPDF2:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ' '.join(page.extract_text() or '' for page in reader.pages)
            return text[:5000] if text.strip() else '[PDF with no extractable text — likely scanned images]'
        elif ext in ('.docx', '.doc') and docx:
            doc = docx.Document(filepath)
            text = '\n'.join(p.text for p in doc.paragraphs)
            return text[:5000] if text.strip() else '[Empty document]'
        elif ext in ('.txt', '.md', '.csv'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()[:5000]
        elif ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp'):
            return f'[Image file: {os.path.basename(filepath)}]'
        else:
            return f'[File: {os.path.basename(filepath)} — type: {ext}]'
    except Exception as e:
        return f'[Error reading file: {str(e)}]'

@app.route('/api/assignments/<aid>/ai-analyze', methods=['POST'])
@token_required
def ai_analyze_submission(aid):
    """Trigger AI analysis of a student's assignment submission."""
    if request.user['role'] not in ('teacher', 'admin'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    data = request.get_json()
    student_uid = data.get('student_uid')
    if not student_uid:
        return jsonify({'success': False, 'message': 'student_uid required'}), 400

    # Get assignment details
    tconn = get_db('teacher')
    assignment = tconn.execute('SELECT * FROM assignments WHERE id = ?', (aid,)).fetchone()
    tconn.close()
    if not assignment:
        return jsonify({'success': False, 'message': 'Assignment not found'}), 404

    # Get submission
    sconn = get_db('student')
    submission = sconn.execute('SELECT * FROM assignment_submissions WHERE assignment_id = ? AND student_uid = ?',
                              (aid, student_uid)).fetchone()
    if not submission:
        sconn.close()
        return jsonify({'success': False, 'message': 'No submission found'}), 404

    # Extract text from submission
    sub_text = extract_text_from_file(submission['file_path']) if submission['file_path'] else '[No file]'

    prompt = f"""You are a senior academic professor and assignment evaluator with expertise in {assignment['subject'] or 'general academics'}.

ASSIGNMENT DETAILS:
- Title: {assignment['title']}
- Description: {assignment['description']}
- Subject: {assignment['subject']}
- Max Score: {assignment['max_score']}

STUDENT SUBMISSION (UID: {student_uid}):
---
{sub_text[:6000]}
---

EVALUATION CRITERIA — analyze each carefully:
1. **Relevance**: Does the submission address the assignment topic/requirements?
2. **Completeness**: Are all parts of the assignment covered?
3. **Accuracy**: Are the facts, concepts, and solutions correct?
4. **Depth**: Does it show understanding beyond surface level?
5. **Clarity**: Is the writing/work clear, organized, and well-structured?
6. **Originality**: Does it show original thinking or just copied content?

Provide your evaluation as a JSON object with these EXACT fields:
{{
  "summary": "A thorough 3-4 sentence summary describing what the student submitted, what approach they took, and overall impression",
  "relevance_score": <0-100 integer. 90-100=perfectly on topic, 70-89=mostly relevant, 50-69=partially relevant, below 50=off topic>,
  "quality_rating": <0-10 integer. 9-10=exceptional, 7-8=good, 5-6=average, 3-4=below average, 1-2=poor>,
  "completeness_score": <0-100 integer indicating how complete the submission is>,
  "accuracy_score": <0-100 integer indicating factual/technical accuracy>,
  "strengths": ["strength 1 with specific example from submission", "strength 2 with detail", "strength 3"],
  "weaknesses": ["specific weakness 1 with suggestion to fix", "weakness 2 with improvement tip", "weakness 3"],
  "key_concepts_covered": ["concept 1", "concept 2", "concept 3"],
  "missing_elements": ["what's missing 1", "what's missing 2"],
  "remarks": "A detailed, constructive 4-6 sentence feedback paragraph. Mention specific parts of the work. Include actionable suggestions for improvement. Be encouraging but honest about gaps.",
  "grade_suggestion": "A+ / A / A- / B+ / B / B- / C+ / C / D / F",
  "suggested_score": <numeric score out of {assignment['max_score']}>
}}

IMPORTANT: Respond with ONLY valid JSON. No markdown code fences, no explanation outside JSON."""

    try:
        client = get_groq()
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert academic evaluator. Always respond with valid JSON only. Never include markdown fences or explanatory text outside the JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        ai_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Remove markdown code fences if present
            if ai_text.startswith('```'):
                ai_text = ai_text.split('\n', 1)[1].rsplit('```', 1)[0]
            ai_result = json.loads(ai_text)
        except json.JSONDecodeError:
            ai_result = {
                'summary': ai_text[:500],
                'relevance_score': 50,
                'quality_rating': 5,
                'strengths': [],
                'weaknesses': [],
                'remarks': ai_text,
                'grade_suggestion': 'N/A'
            }

        # Save to DB
        full_remarks = json.dumps(ai_result)
        sconn.execute('''UPDATE assignment_submissions SET
                        ai_summary = ?, ai_relevance_score = ?, ai_quality_rating = ?,
                        ai_remarks = ?, ai_analyzed = 1
                        WHERE assignment_id = ? AND student_uid = ?''',
                     (ai_result.get('summary', ''),
                      ai_result.get('relevance_score', 0),
                      ai_result.get('quality_rating', 0),
                      full_remarks, aid, student_uid))
        sconn.commit()
        sconn.close()

        return jsonify({'success': True, 'analysis': ai_result})
    except Exception as e:
        sconn.close()
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': f'AI analysis failed: {str(e)}'}), 500

@app.route('/api/assignments/<aid>/submissions', methods=['GET'])
@token_required
def get_submissions_enhanced(aid):
    """Get all submissions with AI analysis data."""
    conn = get_db('student')
    rows = conn.execute('''SELECT s.*, st.name as student_name
                          FROM assignment_submissions s
                          LEFT JOIN students st ON s.student_uid = st.uid
                          WHERE s.assignment_id = ? ORDER BY s.submitted_at DESC''', (aid,)).fetchall()
    conn.close()
    return jsonify({'success': True, 'submissions': [dict(r) for r in rows]})

@app.route('/api/assignments/<aid>/my-submission', methods=['GET'])
@token_required
def get_my_submission(aid):
    """Student gets their own submission + AI feedback."""
    conn = get_db('student')
    row = conn.execute('SELECT * FROM assignment_submissions WHERE assignment_id = ? AND student_uid = ?',
                      (aid, request.user['uid'])).fetchone()
    conn.close()
    if not row:
        return jsonify({'success': True, 'submission': None})
    return jsonify({'success': True, 'submission': dict(row)})

# ── STUDENT ENGAGEMENT TRACKING ────────────────────────────
@app.route('/api/engagement/<session_id>', methods=['GET'])
@token_required
def get_engagement(session_id):
    """Get engagement data for all students in a session."""
    conn = get_db('student')
    rows = conn.execute('''SELECT e.*, s.name as student_name
                          FROM student_engagement e
                          LEFT JOIN students s ON e.student_uid = s.uid
                          WHERE e.session_id = ?''', (session_id,)).fetchall()
    conn.close()
    return jsonify({'success': True, 'engagement': [dict(r) for r in rows]})

@app.route('/api/engagement/student/<uid>', methods=['GET'])
@token_required
def get_student_engagement(uid):
    """Get historical engagement data for a student."""
    conn = get_db('student')
    rows = conn.execute('SELECT * FROM student_engagement WHERE student_uid = ? ORDER BY updated_at DESC LIMIT 50', (uid,)).fetchall()
    conn.close()
    return jsonify({'success': True, 'history': [dict(r) for r in rows]})

@app.route('/api/engagement/hand-raise', methods=['POST'])
@token_required
def log_hand_raise():
    """Teacher logs a hand raise for a student."""
    data = request.get_json()
    student_uid = data.get('student_uid')
    session_id = data.get('session_id')
    conn = get_db('student')
    existing = conn.execute('SELECT * FROM student_engagement WHERE student_uid = ? AND session_id = ?',
                           (student_uid, session_id)).fetchone()
    if existing:
        conn.execute('UPDATE student_engagement SET hand_raises = hand_raises + 1, updated_at = ? WHERE student_uid = ? AND session_id = ?',
                    (datetime.now().isoformat(), student_uid, session_id))
    else:
        conn.execute('INSERT INTO student_engagement (student_uid, session_id, hand_raises) VALUES (?, ?, 1)',
                    (student_uid, session_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/engagement/answer', methods=['POST'])
@token_required
def log_answer():
    """Teacher logs an answer given by a student."""
    data = request.get_json()
    student_uid = data.get('student_uid')
    session_id = data.get('session_id')
    conn = get_db('student')
    existing = conn.execute('SELECT * FROM student_engagement WHERE student_uid = ? AND session_id = ?',
                           (student_uid, session_id)).fetchone()
    if existing:
        conn.execute('UPDATE student_engagement SET answers_given = answers_given + 1, updated_at = ? WHERE student_uid = ? AND session_id = ?',
                    (datetime.now().isoformat(), student_uid, session_id))
    else:
        conn.execute('INSERT INTO student_engagement (student_uid, session_id, answers_given) VALUES (?, ?, 1)',
                    (student_uid, session_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ── STUDENT SUBJECT-WISE PROGRESS ──────────────────────────
@app.route('/api/engagement/student-progress', methods=['GET'])
@token_required
def get_student_progress():
    """Get per-student, per-subject engagement stats for insights."""
    tconn = get_db('teacher')
    sessions = tconn.execute('SELECT id, subject, section, date FROM attendance_sessions ORDER BY date DESC').fetchall()
    tconn.close()
    
    sconn = get_db('student')
    students = sconn.execute('SELECT uid, name, section FROM students').fetchall()
    
    progress = {}
    for st in students:
        uid = st['uid']
        progress[uid] = {'name': st['name'], 'section': st['section'], 'subjects': {}, 'moods': {}}
    
    for sess in sessions:
        sid = sess['id']
        subj = sess['subject']
        engagements = sconn.execute('SELECT * FROM student_engagement WHERE session_id = ?', (sid,)).fetchall()
        for e in engagements:
            uid = e['student_uid']
            if uid not in progress:
                progress[uid] = {'name': uid, 'section': '', 'subjects': {}, 'moods': {}}
            if subj not in progress[uid]['subjects']:
                progress[uid]['subjects'][subj] = {'sessions': 0, 'avg_attention': 0, 'avg_engagement': 0, 'total_hands': 0, 'total_answers': 0, 'moods': []}
            p = progress[uid]['subjects'][subj]
            p['sessions'] += 1
            p['avg_attention'] = int((p['avg_attention'] * (p['sessions']-1) + (e['attention_score'] or 50)) / p['sessions'])
            p['avg_engagement'] = int((p['avg_engagement'] * (p['sessions']-1) + (e['engagement_score'] or 50)) / p['sessions'])
            p['total_hands'] += (e['hand_raises'] or 0)
            p['total_answers'] += (e['answers_given'] or 0)
            m = e['mood'] or 'neutral'
            p['moods'].append(m)
            progress[uid]['moods'][m] = progress[uid]['moods'].get(m, 0) + 1
    
    sconn.close()
    
    # Compute favorite subject per student
    result = []
    for uid, data in progress.items():
        if not data['subjects']:
            continue
        fav = max(data['subjects'].items(), key=lambda x: x[1]['avg_engagement'])
        result.append({
            'uid': uid, 'name': data['name'], 'section': data['section'],
            'subjects': data['subjects'],
            'favorite_subject': fav[0],
            'mood_distribution': data['moods']
        })
    
    return jsonify({'success': True, 'progress': result})

# ── AI SESSION SUGGESTIONS ─────────────────────────────────
@app.route('/api/engagement/<session_id>/ai-suggestions', methods=['GET'])
@token_required
def get_ai_suggestions(session_id):
    """Generate AI-powered teaching suggestions based on session engagement data."""
    teacher_name = request.user.get('name', 'Teacher')
    
    tconn = get_db('teacher')
    session_info = tconn.execute('SELECT * FROM attendance_sessions WHERE id = ?', (session_id,)).fetchone()
    tconn.close()
    
    if not session_info:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    
    sconn = get_db('student')
    engagements = sconn.execute('''SELECT e.student_uid, e.attention_score, e.mood, e.engagement_score, 
        e.hand_raises, e.answers_given, s.name as student_name
        FROM student_engagement e LEFT JOIN students s ON e.student_uid = s.uid
        WHERE e.session_id = ?''', (session_id,)).fetchall()
    sconn.close()
    
    if not engagements:
        return jsonify({'success': True, 'suggestions': 'No engagement data yet. Data auto-populates during live attendance sessions.'})
    
    data_lines = []
    for e in engagements:
        data_lines.append(f"- {e['student_name'] or e['student_uid']}: Attention={e['attention_score']}%, Mood={e['mood']}, Engagement={e['engagement_score']}%, Hands={e['hand_raises']}, Answers={e['answers_given']}")
    
    avg_att = sum(e['attention_score'] or 0 for e in engagements) / max(len(engagements), 1)
    avg_eng = sum(e['engagement_score'] or 0 for e in engagements) / max(len(engagements), 1)
    moods = [e['mood'] for e in engagements if e['mood']]
    dom_mood = max(set(moods), key=moods.count) if moods else 'neutral'
    total_h = sum(e['hand_raises'] or 0 for e in engagements)
    total_a = sum(e['answers_given'] or 0 for e in engagements)
    
    prompt = f"""Analyze this classroom session and give teaching suggestions.

SESSION: {session_info['subject']} | Section: {session_info['section']} | Date: {session_info['date']}
METRICS: Avg Attention={avg_att:.0f}%, Avg Engagement={avg_eng:.0f}%, Class Mood={dom_mood}, Hands={total_h}, Answers={total_a}, Students={len(engagements)}

STUDENTS:
{chr(10).join(data_lines)}

Provide in this format:
**Overall Assessment**: 2 sentences.
**Key Observations**: 3 bullet points.
**Students Needing Attention**: Name + concern + suggestion.
**Teaching Recommendations**: 3 numbered strategies.
**Subject Tip**: One tip for teaching {session_info['subject']} more engagingly."""

    try:
        client = get_groq()
        resp = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[{"role": "system", "content": "You are an expert educational consultant. Be specific and reference actual student data."},
                      {"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=1200
        )
        return jsonify({'success': True, 'suggestions': resp.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ── AI CHATBOT (Groq) ─────────────────────────────────────
@app.route('/api/chatbot/message', methods=['POST'])
@token_required
def chatbot_message():
    """Send a message to the AI chatbot and get a response."""
    if request.user['role'] not in ('teacher', 'admin'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'success': False, 'message': 'Empty message'}), 400

    teacher_uid = request.user['uid']
    teacher_name = request.user['name']

    # Build context from databases
    context_parts = []

    # Timetable
    tconn = get_db('teacher')
    tt = tconn.execute('SELECT * FROM timetable WHERE teacher_uid = ? ORDER BY day_of_week, start_time', (teacher_uid,)).fetchall()
    if tt:
        tt_text = "TEACHER'S TIMETABLE:\n"
        for t in tt:
            tt_text += f"  - {t['day_of_week']}: {t['subject']} ({t['section']}) | {t['start_time']}-{t['end_time']} | Room: {t['room']}\n"
        context_parts.append(tt_text)

    # Recent sessions
    sessions = tconn.execute("SELECT * FROM attendance_sessions WHERE teacher_uid = ? ORDER BY date DESC LIMIT 10", (teacher_uid,)).fetchall()
    if sessions:
        sess_text = "RECENT ATTENDANCE SESSIONS:\n"
        for s in sessions:
            sess_text += f"  - {s['date']}: {s['subject']} ({s['section']}) | Present: {s['total_present']}/{s['total_enrolled']} | Status: {s['status']}\n"
        context_parts.append(sess_text)

    # Assignments
    assignments = tconn.execute('SELECT * FROM assignments WHERE teacher_uid = ? ORDER BY created_at DESC LIMIT 10', (teacher_uid,)).fetchall()
    if assignments:
        assgn_text = "ASSIGNMENTS:\n"
        for a in assignments:
            assgn_text += f"  - [{a['id']}] {a['title']} | Subject: {a['subject']} | Section: {a['section']} | Due: {a['due_date']}\n"
        context_parts.append(assgn_text)

    # Chat history (last 10 messages)
    history = tconn.execute('SELECT role, content FROM chat_messages WHERE teacher_uid = ? ORDER BY created_at DESC LIMIT 10', (teacher_uid,)).fetchall()
    tconn.close()

    # Students
    sconn = get_db('student')
    students = sconn.execute('SELECT uid, name, section, semester, department, email, face_registered FROM students WHERE is_active = 1').fetchall()
    if students:
        stu_text = f"REGISTERED STUDENTS ({len(students)} total):\n"
        for s in students:
            stu_text += f"  - UID: {s['uid']} | Name: {s['name']} | Section: {s['section']} | Sem: {s['semester']} | Dept: {s['department']} | Email: {s['email']} | Face: {'Registered' if s['face_registered'] else 'Not Registered'}\n"
        context_parts.append(stu_text)

    # Attendance records per student (last 30 days)
    attendance = sconn.execute("""SELECT student_uid, subject, section, date, status, presence_percentage 
        FROM attendance ORDER BY date DESC LIMIT 200""").fetchall()
    if attendance:
        att_text = "ATTENDANCE RECORDS (recent):\n"
        for a in attendance:
            att_text += f"  - {a['student_uid']} | {a['date']} | {a['subject']} | {a['section']} | {a['status']} | {a['presence_percentage']}%\n"
        context_parts.append(att_text)

    # Assignment submissions
    submissions = sconn.execute("""SELECT assignment_id, student_uid, file_name, submitted_at, grade, feedback, ai_summary, ai_quality_rating
        FROM assignment_submissions ORDER BY submitted_at DESC LIMIT 50""").fetchall()
    if submissions:
        sub_text = "ASSIGNMENT SUBMISSIONS:\n"
        for s in submissions:
            sub_text += f"  - Assignment {s['assignment_id']} | Student: {s['student_uid']} | File: {s['file_name']} | Submitted: {s['submitted_at']} | Grade: {s['grade'] or 'Pending'} | AI Rating: {s['ai_quality_rating'] or 'Not analyzed'}\n"
        context_parts.append(sub_text)

    # Engagement data
    engagement = sconn.execute("""SELECT student_uid, session_id, attention_score, mood, engagement_score, hand_raises, answers_given 
        FROM student_engagement ORDER BY updated_at DESC LIMIT 100""").fetchall()
    if engagement:
        eng_text = "STUDENT ENGAGEMENT DATA:\n"
        for e in engagement:
            eng_text += f"  - {e['student_uid']} | Session: {e['session_id']} | Attention: {e['attention_score']}% | Mood: {e['mood']} | Engagement: {e['engagement_score']}% | Hands: {e['hand_raises']} | Answers: {e['answers_given']}\n"
        context_parts.append(eng_text)

    sconn.close()

    context = '\n'.join(context_parts)

    system_prompt = f"""You are AttendAI Assistant — an advanced AI teaching assistant for {teacher_name} (UID: {teacher_uid}).

CAPABILITIES:
1. CLASSROOM DATA: Full access to timetable, students, attendance, assignments, submissions, and engagement data.
2. ACADEMIC EXPERT: Can explain any topic, concept, syllabus, or subject matter in detail.
3. LESSON PLANNING: Can suggest lesson plans, teaching strategies, and assessment methods.
4. DATA ANALYSIS: Can analyze attendance trends, student performance, and engagement patterns.
5. REPORT GENERATION: Can create summaries and reports from available data.

FORMATTING RULES:
- Use **bold** for important terms and headings
- Use bullet points for lists
- Use `code` formatting for UIDs and data values
- When showing multiple records, use clean tabular format with | separators
- Be thorough in analysis but concise in language
- When answering academic questions, provide clear, well-structured explanations
- Use appropriate detail level — brief for data queries, detailed for topic explanations
- Today: {datetime.now().strftime('%A, %B %d, %Y')}

CURRENT DATA:
{context}

If data is not available, clearly state what's missing. For academic questions not related to classroom data, answer as a knowledgeable professor would."""

    # Build messages list with history
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for h in reversed(list(history)):
            messages.append({"role": h['role'], "content": h['content']})
    messages.append({"role": "user", "content": user_msg})

    try:
        client = get_groq()
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=2500
        )
        ai_reply = response.choices[0].message.content.strip()

        # Save to chat history
        tconn = get_db('teacher')
        tconn.execute('INSERT INTO chat_messages (teacher_uid, role, content) VALUES (?, ?, ?)',
                     (teacher_uid, 'user', user_msg))
        tconn.execute('INSERT INTO chat_messages (teacher_uid, role, content) VALUES (?, ?, ?)',
                     (teacher_uid, 'assistant', ai_reply))
        tconn.commit()
        tconn.close()

        return jsonify({'success': True, 'reply': ai_reply})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': f'Chatbot error: {str(e)}'}), 500

@app.route('/api/chatbot/history', methods=['GET'])
@token_required
def chatbot_history():
    """Get chat history for current teacher."""
    conn = get_db('teacher')
    rows = conn.execute('SELECT role, content, created_at FROM chat_messages WHERE teacher_uid = ? ORDER BY created_at ASC LIMIT 100',
                       (request.user['uid'],)).fetchall()
    conn.close()
    return jsonify({'success': True, 'messages': [dict(r) for r in rows]})

@app.route('/api/chatbot/clear', methods=['DELETE'])
@token_required
def chatbot_clear():
    """Clear chat history."""
    conn = get_db('teacher')
    conn.execute('DELETE FROM chat_messages WHERE teacher_uid = ?', (request.user['uid'],))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Chat history cleared'})

# ══════════════════════════════════════════════════════════
# ── AI EXAM SYSTEM ENDPOINTS ──────────────────────────────
# ══════════════════════════════════════════════════════════

@app.route('/api/exams', methods=['GET'])
@token_required
def get_exams():
    """List all exams created by this teacher."""
    conn = get_db('teacher')
    rows = conn.execute('SELECT * FROM exams WHERE teacher_uid = ? ORDER BY created_at DESC',
                        (request.user['uid'],)).fetchall()
    conn.close()
    return jsonify({'success': True, 'exams': [dict(r) for r in rows]})


@app.route('/api/exams', methods=['POST'])
@token_required
def create_exam():
    """Create exam and generate MCQ questions using AI."""
    data = request.get_json()
    title = data.get('title', '').strip()
    subject = data.get('subject', '').strip()
    topic = data.get('topic', '').strip()
    course = data.get('course', '')
    section = data.get('section', '')
    difficulty = data.get('difficulty', 'medium')
    num_q = min(int(data.get('num_questions', 10)), 50)
    duration = int(data.get('duration_min', 30))

    if not title or not subject or not topic:
        return jsonify({'success': False, 'message': 'Title, subject, and topic are required'}), 400

    exam_id = str(uuid.uuid4())[:8]

    # Generate questions with AI
    try:
        groq = get_groq()
        prompt = f"""Generate exactly {num_q} multiple choice questions (MCQs) about the topic "{topic}" for the subject "{subject}".
Course/Class: {course or 'General'}
Difficulty Level: {difficulty}

IMPORTANT RULES:
- Each question must have exactly 4 options: A, B, C, D
- Exactly one option must be correct
- Questions should be clear and unambiguous
- Options should be plausible but only one correct
- Include a brief explanation for the correct answer

Return ONLY a JSON array (no markdown, no extra text) in this exact format:
[
  {{
    "question_text": "What is ...?",
    "option_a": "First option",
    "option_b": "Second option",
    "option_c": "Third option",
    "option_d": "Fourth option",
    "correct_option": "A",
    "explanation": "Brief explanation why A is correct"
  }}
]"""

        response = groq.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert exam question generator. Return ONLY valid JSON arrays, no markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        content = response.choices[0].message.content.strip()
        # Clean markdown formatting if present
        if content.startswith('```'):
            content = content.split('\n', 1)[1] if '\n' in content else content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

        questions = json.loads(content)
        if not isinstance(questions, list):
            questions = [questions]
        questions = questions[:num_q]

    except Exception as e:
        print(f"[EXAM] AI generation error: {e}")
        return jsonify({'success': False, 'message': f'AI generation failed: {str(e)}'}), 500

    # Save exam to database
    try:
        conn = get_db('teacher')
        conn.execute('''INSERT INTO exams (id, teacher_uid, title, subject, topic, course, section, difficulty, num_questions, duration_min, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft')''',
                    (exam_id, request.user['uid'], title, subject, topic, course, section, difficulty, len(questions), duration))

        for i, q in enumerate(questions):
            conn.execute('''INSERT INTO exam_questions (exam_id, question_num, question_text, option_a, option_b, option_c, option_d, correct_option, explanation)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (exam_id, i + 1, q.get('question_text', ''), q.get('option_a', ''), q.get('option_b', ''),
                         q.get('option_c', ''), q.get('option_d', ''), q.get('correct_option', 'A'), q.get('explanation', '')))

        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'exam_id': exam_id,
            'exam': {'id': exam_id, 'title': title, 'subject': subject, 'topic': topic, 'course': course,
                     'section': section, 'difficulty': difficulty, 'num_questions': len(questions),
                     'duration_min': duration, 'status': 'draft'},
            'questions': questions
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/exams/<exam_id>', methods=['GET'])
@token_required
def get_exam(exam_id):
    """Get exam details. Teachers see answers, students don't."""
    conn = get_db('teacher')
    exam = conn.execute('SELECT * FROM exams WHERE id = ?', (exam_id,)).fetchone()
    if not exam:
        conn.close()
        return jsonify({'success': False, 'message': 'Exam not found'}), 404

    questions = conn.execute('SELECT * FROM exam_questions WHERE exam_id = ? ORDER BY question_num', (exam_id,)).fetchall()
    conn.close()

    exam_dict = dict(exam)
    q_list = [dict(q) for q in questions]

    # If student, remove correct answers
    if request.user.get('role') == 'student':
        for q in q_list:
            q.pop('correct_option', None)
            q.pop('explanation', None)

    return jsonify({'success': True, 'exam': exam_dict, 'questions': q_list})


@app.route('/api/exams/<exam_id>/publish', methods=['POST'])
@token_required
def publish_exam(exam_id):
    """Publish exam so students can take it."""
    conn = get_db('teacher')
    exam = conn.execute('SELECT * FROM exams WHERE id = ? AND teacher_uid = ?', (exam_id, request.user['uid'])).fetchone()
    if not exam:
        conn.close()
        return jsonify({'success': False, 'message': 'Exam not found'}), 404
    conn.execute("UPDATE exams SET status = 'published', published_at = ? WHERE id = ?",
                (datetime.now().isoformat(), exam_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Exam published successfully'})


@app.route('/api/exams/<exam_id>/close', methods=['POST'])
@token_required
def close_exam(exam_id):
    """Close exam - no more submissions."""
    conn = get_db('teacher')
    conn.execute("UPDATE exams SET status = 'closed', closed_at = ? WHERE id = ?",
                (datetime.now().isoformat(), exam_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Exam closed'})


@app.route('/api/exams/<exam_id>/delete', methods=['DELETE'])
@token_required
def delete_exam(exam_id):
    """Delete exam and all related data."""
    conn = get_db('teacher')
    conn.execute('DELETE FROM exam_questions WHERE exam_id = ?', (exam_id,))
    conn.execute('DELETE FROM exam_results WHERE exam_id = ?', (exam_id,))
    conn.execute('DELETE FROM exams WHERE id = ? AND teacher_uid = ?', (exam_id, request.user['uid']))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Exam deleted'})


@app.route('/api/exams/available', methods=['GET'])
@token_required
def get_available_exams():
    """Get published exams available for the student."""
    student_uid = request.user['uid']
    # Get student section
    sconn = get_db('student')
    student = sconn.execute('SELECT section FROM students WHERE uid = ?', (student_uid,)).fetchone()
    student_section = student['section'] if student else ''
    sconn.close()

    conn = get_db('teacher')
    exams = conn.execute("SELECT * FROM exams WHERE status = 'published' ORDER BY published_at DESC").fetchall()
    conn.close()

    # Filter by section if student has one
    result = []
    for e in exams:
        e_dict = dict(e)
        if e_dict.get('section') and student_section and e_dict['section'].lower() != student_section.lower():
            continue

        # Check if already submitted
        sconn2 = get_db('student')
        sub = sconn2.execute('SELECT * FROM student_exams WHERE exam_id = ? AND student_uid = ?',
                            (e_dict['id'], student_uid)).fetchone()
        sconn2.close()
        if sub:
            e_dict['status'] = 'submitted'
            e_dict['score'] = sub['score']
            e_dict['total'] = sub['total']
            e_dict['percentage'] = sub['percentage']
        result.append(e_dict)

    return jsonify({'success': True, 'exams': result})


@app.route('/api/exams/<exam_id>/submit', methods=['POST'])
@token_required
def submit_exam(exam_id):
    """Submit exam answers with proctoring data."""
    data = request.get_json()
    answers = data.get('answers', {})
    time_taken = int(data.get('time_taken_sec', 0))
    proctor = data.get('proctor_summary', {})

    student_uid = request.user['uid']
    student_name = request.user.get('name', '')

    # Check if already submitted
    sconn = get_db('student')
    existing = sconn.execute('SELECT id FROM student_exams WHERE exam_id = ? AND student_uid = ?',
                            (exam_id, student_uid)).fetchone()
    if existing:
        sconn.close()
        return jsonify({'success': False, 'message': 'Already submitted'}), 409

    # Get correct answers
    tconn = get_db('teacher')
    questions = tconn.execute('SELECT question_num, correct_option FROM exam_questions WHERE exam_id = ? ORDER BY question_num',
                             (exam_id,)).fetchall()
    exam = tconn.execute('SELECT * FROM exams WHERE id = ?', (exam_id,)).fetchone()

    if not exam:
        tconn.close()
        sconn.close()
        return jsonify({'success': False, 'message': 'Exam not found'}), 404

    # Grade
    score = 0
    total = len(questions)
    for q in questions:
        student_ans = answers.get(str(q['question_num'] - 1), '')
        if student_ans.upper() == q['correct_option'].upper():
            score += 1

    percentage = round((score / total * 100), 1) if total > 0 else 0

    # Calculate risk level from proctoring data
    head_turns = proctor.get('headTurns', 0)
    eye_alerts = proctor.get('eyeAlerts', 0)
    tab_switches = proctor.get('tabSwitches', 0)
    face_not_visible = proctor.get('faceNotVisible', 0)

    risk_score = tab_switches * 3 + head_turns * 1 + face_not_visible * 2 + eye_alerts * 0.5
    risk_level = 'low' if risk_score < 5 else 'medium' if risk_score < 15 else 'high'

    # AI analysis of proctoring
    ai_analysis = ''
    try:
        if risk_score > 0:
            groq = get_groq()
            analysis_prompt = f"""Analyze this student's exam proctoring data and provide a brief integrity assessment (2-3 sentences):
- Head turns away from screen: {head_turns}
- Eye movement anomalies: {eye_alerts}
- Tab/window switches: {tab_switches}
- Face not visible moments: {face_not_visible}
- Exam duration: {time_taken} seconds
- Score: {score}/{total} ({percentage}%)
- Risk level: {risk_level}

Provide a professional assessment of the student's exam behavior."""

            ai_resp = groq.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=[{"role": "system", "content": "You are an exam integrity analyst. Be concise and professional."},
                          {"role": "user", "content": analysis_prompt}],
                temperature=0.3, max_tokens=200
            )
            ai_analysis = ai_resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[EXAM] AI analysis error: {e}")

    # Save result in teacher DB
    tconn.execute('''INSERT OR REPLACE INTO exam_results
                    (exam_id, student_uid, student_name, score, total, percentage, answers, time_taken_sec,
                     proctor_summary, head_turn_count, eye_movement_alerts, tab_switch_count,
                     face_not_visible_count, risk_level, ai_analysis)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (exam_id, student_uid, student_name, score, total, percentage,
                  json.dumps(answers), time_taken, json.dumps(proctor),
                  head_turns, eye_alerts, tab_switches, face_not_visible, risk_level, ai_analysis))
    tconn.commit()
    tconn.close()

    # Save in student DB
    sconn.execute('''INSERT OR REPLACE INTO student_exams (exam_id, student_uid, status, score, total, percentage, submitted_at)
                    VALUES (?, ?, 'submitted', ?, ?, ?, ?)''',
                 (exam_id, student_uid, score, total, percentage, datetime.now().isoformat()))
    sconn.commit()
    sconn.close()

    return jsonify({'success': True, 'score': score, 'total': total, 'percentage': percentage,
                    'risk_level': risk_level})


@app.route('/api/exams/<exam_id>/results', methods=['GET'])
@token_required
def get_exam_results(exam_id):
    """Get all student results for an exam."""
    conn = get_db('teacher')
    exam = conn.execute('SELECT * FROM exams WHERE id = ?', (exam_id,)).fetchone()
    results = conn.execute('SELECT * FROM exam_results WHERE exam_id = ? ORDER BY submitted_at DESC',
                          (exam_id,)).fetchall()
    conn.close()

    if not exam:
        return jsonify({'success': False, 'message': 'Exam not found'}), 404

    return jsonify({'success': True, 'exam': dict(exam), 'results': [dict(r) for r in results]})


# ── Create Default Admin ───────────────────────────────────
def create_default_admin():
    conn = get_db('admin')
    admin = conn.execute('SELECT uid FROM admins LIMIT 1').fetchone()
    if not admin:
        pw = hash_password('admin123')
        conn.execute('INSERT INTO admins (uid, name, email, password_hash) VALUES (?, ?, ?, ?)',
                    ('ADMIN001', 'System Admin', 'admin@attendai.com', pw))
        conn.commit()
        print("[ADMIN] Default admin created: ADMIN001 / admin123")
    conn.close()

# ── Main ───────────────────────────────────────────────────
if __name__ == '__main__':
    create_default_admin()
    print("\n" + "="*50)
    print("  AttendAI Server Starting...")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

