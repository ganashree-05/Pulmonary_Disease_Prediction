# app.py
import os
import json
import sqlite3
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, send_from_directory, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# config and utils (make sure these exist as separate files)
try:
    import config
except Exception:
    raise RuntimeError("Missing config.py — please add config.py as instructed.")

from config import UPLOAD_DIR, REPORT_DIR, DATA_DIR, MODEL_PATH, DB_PATH, SECRET_KEY

# utils: we expect init_db(), save_submission(...), generate_pdf(data, lang)
# If user provided utils.py earlier, import; else we will implement minimal helpers here.
try:
    from utils import init_db as utils_init_db, save_submission as utils_save_submission, generate_pdf as utils_generate_pdf
    HAVE_UTILS = True
except Exception:
    HAVE_UTILS = False

# ---------- ML model ----------
from xray_model import load_model_safe
from xray_model import load_model_safe, predict_image_safe




# Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY or "replace-this-secret"

# Ensure folders exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# --------- DB helpers (sqlite) ----------
def init_db():
    """Create tables users and submissions if they don't exist."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # users: id, username (unique), password_hash, is_admin (0/1)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            is_admin INTEGER DEFAULT 0
        )
    """)
    # submissions: keep what you had + pdf_report
    cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            user_id INTEGER,
            username TEXT,
            lang TEXT,
            age_group TEXT,
            general_yes TEXT,
            age_yes TEXT,
            risk TEXT,
            suggestions TEXT,
            xray_filename TEXT,
            ml_pred TEXT,
            ml_conf REAL,
            pdf_report TEXT
        )
    """)
    con.commit()
    # ensure default admin exists
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if not cur.fetchone():
        pw_hash = generate_password_hash("admin123")
        admin_email = "admin@example.com"  # <-- set your admin email here
        cur.execute(
            "INSERT INTO users (username, password_hash, email, is_admin) VALUES (?, ?, ?, ?)",
            ("admin", pw_hash, admin_email, 1)
        )
        con.commit()
    con.close()


# If utils provided init_db, call that instead (but only if user didn't want our DB)
if HAVE_UTILS:
    try:
        utils_init_db()
    except Exception:
        # still ensure our schema
        init_db()
else:
    init_db()

def db_get_user_by_username(username):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, username, password_hash, is_admin FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    con.close()
    return row  # (id, username, password_hash, is_admin) or None

def db_create_user(username, password, is_admin=0):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), is_admin))
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()

def db_save_submission(record):
    """
    record: dict with keys (user_id, username, lang, age_group, general_yes, age_yes, risk, suggestions, xray_filename, ml_pred, ml_conf, pdf_report)
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO submissions (ts, user_id, username, lang, age_group, general_yes, age_yes, risk, suggestions,
                                 xray_filename, ml_pred, ml_conf, pdf_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        record.get("user_id"),
        record.get("username"),
        record.get("lang"),
        record.get("age_group"),
        json.dumps(sorted(list(record.get("general_yes", [])))),
        json.dumps(sorted(list(record.get("age_yes", [])))),
        record.get("risk"),
        record.get("suggestions"),
        record.get("xray_filename"),
        record.get("ml_pred"),
        record.get("ml_conf"),
        record.get("pdf_report")
    ))
    con.commit()
    lid = cur.lastrowid
    con.close()
    return lid

def db_get_submissions(limit=200):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, ts, user_id, username, lang, age_group, risk, pdf_report FROM submissions ORDER BY ts DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    return rows

def db_get_submission(submission_id):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT * FROM submissions WHERE id=?", (submission_id,))
    row = cur.fetchone()
    con.close()
    return row

def db_update_submission_pdf(submission_id, pdf_name):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE submissions SET pdf_report=? WHERE id=?", (pdf_name, submission_id))
    con.commit()
    con.close()

# ---------- Questions store (admin editable) ----------
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.json")
DEFAULT_QUESTIONS = {
    "general": [
        ["cough_days", {"en":"Cough lasting more than a few days?","hi":"क्या आपको कई दिनों से खांसी है?","kn":"ಕೆಲವು ದಿನಗಳಿಗೂ ಅಧಿಕವಾಗಿ ಕೆಮ್ಮು ಇದೆಯೇ?"}],
        ["cough_mucus", {"en":"Is your cough producing mucus/phlegm?","hi":"क्या खांसी में बलगम/कफ आ रहा है?","kn":"ಕೆಮ್ಮಿನೊಂದಿಗೆ ಶ್ಲೇಷ್ಮ/ಕಫ್ ಬರುತ್ತಿದೆಯೇ?"}],
        ["fever_high", {"en":"High fever (above 38°C / 100.4°F)?","hi":"उच्च बुखार (38°C / 100.4°F से अधिक)?","kn":"ಹೆಚ್ಚಿನ ಜ್ವರ (38°C / 100.4°F ಮೇಲ್ಪಟ್ಟು)?"}],
        ["chills", {"en":"Chills or shivering?","hi":"कंपकंपी या ठिठुरन?","kn":"ನಡುಕು ಅಥವಾ ತಡಕಾಟ ಇದೆಯೇ?"}],
        ["chest_pain", {"en":"Chest pain when breathing or coughing?","hi":"सांस लेने या खांसने पर सीने में दर्द?","kn":"ಉಸಿರಾಡುವಾಗ/ಕೆಮ್ಮುವಾಗ ಎದೆನೋವು?"}],
        ["short_breath_rest", {"en":"Shortness of breath (even at rest)?","hi":"आराम करते समय भी सांस फूलना?","kn":"ಆರಾಮದಲ್ಲಿಯೂ ಉಸಿರಾಟದಲ್ಲಿ ತೊಂದರೆ?"}],
        ["fatigue", {"en":"Unusual fatigue or weakness?","hi":"असामान्य थकान या कमजोरी?","kn":"ಅಸಾಮಾನ್ಯ ದೌರ್ಬಲ್ಯ/ದೌರ್ಬಲ್ಯ?"}],
        ["loss_appetite", {"en":"Loss of appetite recently?","hi":"क्या भूख कम लग रही है?","kn":"ಇತ್ತೀಚೆಗೆ ಊಟದ ಆಸಕ್ತಿ ಕಡಿಮೆಯೇ?"}],
        ["headache", {"en":"Headache along with fever?","hi":"बुखार के साथ सिरदर्द?","kn":"ಜ್ವರದೊಂದಿಗೆ ತಲೆನೋವು?"}],
        ["dry_cough", {"en":"Dry cough (without mucus)?","hi":"सूखी खांसी (बिना बलगम)?","kn":"ಒಣ ಕೆಮ್ಮು (ಶ್ಲೇಷ್ಮ ಇಲ್ಲದೆ)?"}],
        ["breath_back_pain", {"en":"Pain in shoulders/back while breathing?","hi":"सांस लेते समय कंधे/पीठ में दर्द?","kn":"ಉಸಿರಾಡುವಾಗ ಭುಜ/ಬೆನ್ನು ನೋವು?"}],
        ["cold_hands_feet", {"en":"Cold hands or feet?","hi":"हाथ/पैर ठंडे रहते हैं?","kn":"ಕೈ/ಕಾಲುಗಳು ಅಸಾಮಾನ್ಯವಾಗಿ ಚಳಿಯಾಗಿವೆಯೇ?"}],
        ["confusion", {"en":"Trouble concentrating or confusion?","hi":"ध्यान केंद्रित करने में कठिनाई या भ्रम?","kn":"ಗಮನ ಕೇಂದ್ರೀಕರಿಸುವಲ್ಲಿ ತೊಂದರೆ ಅಥವಾ ಗೊಂದಲ?"}],
        ["rapid_breath_heart", {"en":"Rapid breathing or racing heartbeat?","hi":"तेज़ सांसें या धड़कन तेज़?","kn":"ಶೀಘ್ರ ಉಸಿರಾಟ ಅಥವಾ ಹೃದಯಬಡಿತ ವೇಗವಾಗಿದೆಯೇ?"}],
        ["post_flu", {"en":"Symptoms started after a recent cold/flu?","hi":"हाल की सर्दी/फ्लू के बाद लक्षण?","kn":"ಇತ್ತೀಚಿನ ಜಲದೋಷ/ಫ್ಲೂ ನಂತರ ಲಕ್ಷಣಗಳು ಆರಂಭವಾದವೆಯೇ?"}],
        ["blue_lips", {"en":"Bluish lips or fingernails?","hi":"होठ/उंगलियां नीली पड़ना?","kn":"ತುಟಿ/ನಖಗಳು ನೀಲಿಮೆಯಾಯಿತೇ?"}]
    ],
    "child": [
        ["child_fast_breath", {"en":"Child breathing faster than normal?","hi":"क्या बच्चा सामान्य से तेज़ सांस ले रहा है?","kn":"ಮಗು ಸಾಮಾನ್ಯಕ್ಕಿಂತ ವೇಗವಾಗಿ ಉಸಿರಾಡುತ್ತಿದೆಯೇ?"}],
        ["child_nasal_flaring", {"en":"Nasal flaring while breathing?","hi":"सांस लेते समय नथुने फूलना?","kn":"ಉಸಿರಾಟದ ವೇಳೆ ಮೂಗಿನ ರಂಧ್ರಗಳು ಅಗಲುತ್ತಿವೆಯೇ?"}],
        ["child_blue_skin", {"en":"Bluish lips/skin in child?","hi":"बच्चे की त्वचा/होठ नीले पड़ना?","kn":"ಮಗುವಿನ ತುಟಿ/ಚರ್ಮ ನೀಲಿ ಬಣ್ಣದಲ್ಲಿದೆಯೇ?"}],
        ["child_irritable", {"en":"Unusually irritable or fussy?","hi":"बच्चा असामान्य रूप से चिड़चिड़ा/रोता है?","kn":"ಮಗು ಅಸಾಮಾನ್ಯವಾಗಿ ಕಿರಿಕಿರಿ/ಅಳುವುದು?"}],
        ["child_poor_feeding", {"en":"Stopped feeding/eating well?","hi":"खाना/दूध कम पीना?","kn":"ಹಾಲು/ಆಹಾರ ಕಡಿಮೆ ಸೇವಿಸುತ್ತಿದೆಯೇ?"}],
        ["child_apnea", {"en":"Pauses in breathing (apnea)?","hi":"सांस रुक-रुक कर आना (एप्निया)?","kn":"ಉಸಿರಾಟ ಕೆಲವು ಹೊತ್ತು ನಿಲ್ಲುತ್ತಿದೆಯೇ (ಅಪ್ನಿಯಾ)?"}],
        ["child_vomit_fever", {"en":"Vomiting during fever?","hi":"बुखार में उल्टी?","kn":"ಜ್ವರದಲ್ಲಿ ವಾಂತಿ ಆಗುತ್ತಿದೆಯೇ?"}],
        ["child_oversleep", {"en":"Sleeping much more or hard to wake?","hi":"बहुत ज्यादा सोना/जगाना मुश्किल?","kn":"ಅತ್ಯಧಿಕ ನಿದ್ರೆ/ಎಬ್ಬಿಸಲು ಕಷ್ಟವಾ?"}],
        ["child_rash", {"en":"Rash after fever?","hi":"बुखार के बाद चकत्ते?","kn":"ಜ್ವರದ ನಂತರ ಚರ್ಮದ ದದ್ದು?"}]
    ],
    "adult": [
        ["adult_sweats", {"en":"Excessive sweating with fever?","hi":"बुखार के साथ अत्यधिक पसीना?","kn":"ಜ್ವರದೊಂದಿಗೆ ಅಧಿಕ ಬೆವರು?"}],
        ["adult_tight_chest", {"en":"Tightness/pressure in chest?","hi":"सीने में जकड़न/दबाव?","kn":"ಎದೆಯಲ್ಲಿ ಜಡಿತ/ಒತ್ತಡ?"}],
        ["adult_sore_throat", {"en":"Sore throat with cough?","hi":"खांसी के साथ गले में खराश?","kn":"ಕೆಮ್ಮಿನೊಂದಿಗೆ ಗಂಟಲು ನೋವು?"}],
        ["adult_aches", {"en":"Muscle/joint pain?","hi":"मांसपेशियों/जोड़ों में दर्द?","kn":"ಸ್ನಾಯು/ಸಂಧಿ ನೋವು?"}],
        ["adult_dizzy", {"en":"Feeling dizzy?","hi":"चक्कर आना?","kn":"ತಲೆ ಸುತ್ತುವುದು?"}],
        ["adult_orthopnea", {"en":"Breathlessness when lying down?","hi":"लेटने पर सांस फूलना?","kn":"ಬಿದ್ದುಕೊಂಡಾಗ ಉಸಿರಾಟ ಕಷ್ಟವಾಗುತ್ತಿದೆಯೇ?"}],
        ["adult_wheeze", {"en":"Wheezing sound while breathing?","hi":"सांस में घरघराहट?","kn":"ಉಸಿರಾಟದ ವೇಳೆ ಘರ್ಘರ ಶಬ್ದ (ವೀಸ್)?"}]
    ],
    "senior": [
        ["senior_confusion", {"en":"More confused or disoriented than usual?","hi":"सामान्य से अधिक उलझन/भ्रम?","kn":"ಸಾಮಾನ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚು ಗೊಂದಲ/ತಪ್ಪು ದಿಕ್ಕು?"}],
        ["senior_low_temp", {"en":"Low body temperature instead of fever?","hi":"बुखार की जगह तापमान कम?","kn":"ಜ್ವರ ಬದಲು ದೇಹದ ತಾಪಮಾನ ಕಡಿಮೆ?"}],
        ["senior_feel_cold", {"en":"Feeling unusually cold in warm room?","hi":"गर्म कमरे में भी असामान्य रूप से ठंड लगना?","kn":"ಬೆಚ್ಚಗಿನ ಕೊಠಡಿಯಲ್ಲಿ ಕೂಡ ಅಸಹಜವಾಗಿ ಚಳಿ ಅನಿಸುತಿದೆಯೇ?"}],
        ["senior_bp_drop", {"en":"Sudden drop in blood pressure?","hi":"रक्तचाप अचानक कम होना?","kn":"ರಕ್ತದ ಒತ್ತಡದಲ್ಲಿ ಆಕस್ಮಿಕ ಇಳಿಕೆ?"}],
        ["senior_fall", {"en":"Recent fall due to weakness/dizziness?","hi":"कमजोरी/चक्कर से हाल में गिरना?","kn":"ದೌರ್ಬಲ್ಯ/ತಲೆಸುತ್ತಿನಿಂದ ಇತ್ತೀಚೆಗೆ ಬಿದ್ದಿರಾ?"}],
        ["senior_apathy", {"en":"Loss of interest in daily activities?","hi":"दैनिक कार्यों में रुचि कम?","kn":"ದೈನಂದಿನ ಕಾರ್ಯಗಳಲ್ಲಿ ಆಸಕ್ತಿ ಕಡಿಮೆಯೇ?"}],
        ["senior_urine_change", {"en":"Change in urination frequency/difficulty?","hi":"पेशाब की आदत में बदलाव/कठिनाई?","kn":"ಮೂತ್ರ ವಿಸರ್ಜನೆ ಪ್ರಮಾಣ/ಕಷ್ಟದಲ್ಲಿ ಬದಲಾವಣೆ?"}]
    ]
}

def load_questions():
    if not os.path.exists(QUESTIONS_PATH):
        with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_QUESTIONS, f, ensure_ascii=False, indent=2)
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_questions(qobj):
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(qobj, f, ensure_ascii=False, indent=2)



# ---------- Decorators ----------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please login.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access required.", "danger")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return wrapper

# ---------- Routes ----------
@app.route("/")
def home():
    lang = session.get("lang", "en")
    return render_template("home.html", langs=["en","hi","kn"], lang=lang)

# ----- AUTH -----
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        if not username or not password:
            flash("Provide username and password", "warning")
            return render_template("register.html")
        success = db_create_user(username, password, is_admin=0)
        if not success:
            flash("Username already exists", "danger")
            return render_template("register.html")
        flash("Registered. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        row = db_get_user_by_username(username)
        if row and check_password_hash(row[2], password):
            session["user_id"] = row[0]
            session["username"] = row[1]
            session["is_admin"] = bool(row[3])
            flash("Login successful", "success")
            return redirect(url_for("dashboard") if not row[3] else url_for("admin_dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("home"))

# ----- Dashboard (user) -----
@app.route("/dashboard")
@login_required
def dashboard():
    uid = session.get("user_id")
    # show user's submissions
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, ts, age_group, risk, pdf_report FROM submissions WHERE user_id=? ORDER BY ts DESC", (uid,))
    rows = cur.fetchall()
    con.close()
    return render_template("dashboard.html", submissions=rows)

# ----- Survey flow -----
@app.route("/set-lang", methods=["POST"])
def set_lang():
    lang = request.form.get("lang","en")
    session["lang"] = lang
    return redirect(url_for("general"))

@app.route("/general", methods=["GET","POST"])
@login_required
def general():
    lang = session.get("lang","en")
    questions = load_questions().get("general", [])
    if request.method == "POST":
        yes_keys = set(request.form.getlist("general_yes"))
        session["general_yes"] = list(yes_keys)
        return redirect(url_for("age_select"))
    return render_template("general.html", GENERAL_Q=questions, lang=lang)

@app.route("/age-select", methods=["GET","POST"])
@login_required
def age_select():
    lang = session.get("lang","en")
    if request.method == "POST":
        age_group = request.form.get("age_group")
        if age_group not in ("child","adult","senior"):
            flash("Please select an age group.", "warning")
            return redirect(url_for("age_select"))
        session["age_group"] = age_group
        return redirect(url_for("age_survey"))
    return render_template("age_select.html", lang=lang)

@app.route("/age-survey", methods=["GET","POST"])
@login_required
def age_survey():
    lang = session.get("lang","en")
    age_group = session.get("age_group")
    if not age_group:
        return redirect(url_for("age_select"))
    questions = load_questions().get(age_group, [])
    if request.method == "POST":
        age_yes = set(request.form.getlist("age_yes"))
        session["age_yes"] = list(age_yes)

        general_yes = set(session.get("general_yes", []))
        # Reuse assess_risk logic similar to earlier code:
        # use same rules as provided originally:
        def assess_risk_local(general_yes_set, age_group_local, age_yes_set):
            CRITICAL_GENERAL = {"blue_lips", "short_breath_rest"}
            CRITICAL_CHILD = {"child_apnea", "child_blue_skin", "child_fast_breath"}
            CRITICAL_SENIOR = {"senior_confusion"}
            if (CRITICAL_GENERAL & general_yes_set) or (age_group_local=="child" and (CRITICAL_CHILD & age_yes_set)) or (age_group_local=="senior" and (CRITICAL_SENIOR & age_yes_set)):
                return "high"
            core = {"fever_high", "cough_mucus", "chest_pain", "rapid_breath_heart"}
            if len(core & general_yes_set) >= 2:
                return "moderate"
            if len(general_yes_set) >= 6 or ("cough_days" in general_yes_set and len(general_yes_set) >=4):
                return "moderate"
            return "low"

        risk = assess_risk_local(general_yes, age_group, age_yes)
        session["risk"] = risk

        # suggestions
        def risk_suggestions_local(risk_level, langp):
            txts = {
                "en":{
                    "low":"Low risk at this time. Rest, hydrate, and monitor for 48–72 hours. If symptoms persist/worsen, see a clinician.",
                    "moderate":"Possible pneumonia. Please consult a clinician within 24–48 hours. A chest X-ray may be helpful.",
                    "high":"High risk for pneumonia. Upload a recent chest X-ray and seek medical care as soon as possible."
                },
                "hi":{
                    "low":"वर्तमान में जोखिम कम है। आराम करें, पानी पिएँ, 48–72 घंटे लक्षण देखें। न सुधरे/बढ़ें तो डॉक्टर से मिलें।",
                    "medium":"संभावित निमोनिया। 24–48 घंटे में डॉक्टर से मिलें। छाती का एक्स-रे सहायक हो सकता है।",
                    "high":"निमोनिया का उच्च जोखिम। हाल का छाती एक्स-रे अपलोड करें और जल्द चिकित्सा सहायता लें।"
                },
                "kn":{
                    "low":"ಪ್ರಸ್ತುತ ಅಪಾಯ ಕಡಿಮೆ. ವಿಶ್ರಾಂತಿ ಮಾಡಿ, ನೀರು ಕುಡಿ, 48–72 ಗಂಟೆ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ. ಹೆಚ್ಚಾದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
                    "medium":"ಸಾಧ್ಯವಾದ ನಿಮೋನಿಯಾ. 24–48 ಗಂಟೆಗಳಲ್ಲಿ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ. ಛಾತಿ ಎಕ್ಸ್-ರೆ ಸಹಾಯಕವಾಗಬಹುದು.",
                    "high":"ನಿಮೋನಿಯಾದ ಹೆಚ್ಚಿನ ಅಪಾಯ. ಇತ್ತೀಚಿನ ಛಾತಿ ಎಕ್ಸ್-ರೆ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಶೀಘ್ರ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಪಡೆಯಿರಿ."
                }
            }
            if langp not in txts: langp = "en"
            return txts[langp][risk_level]
        suggestion = risk_suggestions_local(risk, lang)
        session["suggestion"] = suggestion

        # prepare and save submission row (pdf not yet)
        record = {
            "user_id": session.get("user_id"),
            "username": session.get("username"),
            "lang": lang,
            "age_group": age_group,
            "general_yes": general_yes,
            "age_yes": age_yes,
            "risk": risk,
            "suggestions": suggestion,
            "xray_filename": None,
            "ml_pred": None,
            "ml_conf": None,
            "pdf_report": None
        }
        sub_id = db_save_submission(record)
        session["last_submission_id"] = sub_id

        return redirect(url_for("result"))
    return render_template("age_survey.html", QUESTIONS=questions, lang=lang)

@app.route("/result", methods=["GET","POST"])
@login_required
def result():
    # Collect data from session
    survey = {
        "id": session.get("last_submission_id"),
        "risk": session.get("risk"),
        "suggestion": session.get("suggestion"),
        "age_group": session.get("age_group"),
        "general_yes": session.get("general_yes", []),
        "age_yes": session.get("age_yes", []),
        "xray_uploaded": session.get("xray_filename"),
        "xray_prediction": session.get("ml_pred"),
        "ml_conf": session.get("ml_conf"),
    }

    if not survey["risk"]:
        return redirect(url_for("general"))

    if request.method == "POST":
        # generate PDF (as before)
        data = survey.copy()
        # you can also add general_texts, age_texts etc. if needed
        pdf_path = generate_pdf_local(data, session.get("lang","en"))
        if survey["id"]:
            db_update_submission_pdf(survey["id"], os.path.basename(pdf_path))
        return send_file(pdf_path, as_attachment=True)

    return render_template("result.html", survey=survey)



# ---------- ML model ----------
from xray_model import load_model_safe, predict_image_safe

ML_MODEL = None
try:
    ML_MODEL = load_model_safe(MODEL_PATH)
    print("ML model loaded successfully.")
except Exception as e:
    ML_MODEL = None
    print("ML model failed to load:", e)

# ---------- Upload X-ray route ----------
@app.route("/upload_xray", methods=["POST"])
@login_required
def upload_xray():
    if "xray_file" not in request.files or request.files["xray_file"].filename == "":
        flash("No file selected.", "warning")
        return redirect(url_for("result"))

    if ML_MODEL is None:
        flash("ML model not loaded. Cannot predict X-ray.", "danger")
        return redirect(url_for("result"))

    file = request.files["xray_file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        ml_pred, ml_conf = predict_image_safe(ML_MODEL, filepath)
        session["xray_filename"] = filename
        session["ml_pred"] = ml_pred
        session["ml_conf"] = ml_conf
        flash(f"X-ray predicted: {ml_pred} ({ml_conf}%)", "success")

        # Update submission in DB
        sub_id = session.get("last_submission_id")
        if sub_id:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("UPDATE submissions SET xray_filename=?, ml_pred=?, ml_conf=? WHERE id=?",
                        (filename, ml_pred, ml_conf, sub_id))
            con.commit()
            con.close()
    except Exception as e:
        flash(f"Prediction failed: {e}", "danger")

    return redirect(url_for("result"))

# ---------- Result route ----------
@app.route("/result", methods=["GET","POST"])
@login_required
def result_page():  # renamed function to avoid duplicates
    sub_id = session.get("last_submission_id")
    if not sub_id:
        return redirect(url_for("general"))

    # Load submission from DB
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT risk, suggestions, xray_filename, ml_pred, ml_conf FROM submissions WHERE id=?",
        (sub_id,)
    )
    row = cur.fetchone()
    con.close()

    survey = {
        "id": sub_id,
        "risk": row[0] if row else None,
        "suggestion": row[1] if row else None,
        "xray_filename": row[2] if row else None,
        "xray_prediction": row[3] if row else None,
        "ml_conf": row[4] if row else None
    }

    if request.method == "POST":
        # Generate PDF
        data = survey.copy()
        pdf_path = generate_pdf_local(data, session.get("lang","en"))
        if survey["id"]:
            db_update_submission_pdf(survey["id"], os.path.basename(pdf_path))
        return send_file(pdf_path, as_attachment=True)

    return render_template("result.html", survey=survey)

#---------- Admin area ----------
@app.route("/admin")
@admin_required
def admin_dashboard():
    subs = db_get_submissions(500)
    return render_template("admin_dashboard.html", submissions=subs)

@app.route("/admin/download/<int:submission_id>")
@admin_required
def admin_download(submission_id):
    row = db_get_submission(submission_id)
    if not row:
        flash("Not found.", "warning"); return redirect(url_for("admin_dashboard"))
    pdf_name = row[-1]  # pdf_report is last column
    if not pdf_name:
        flash("No PDF attached.", "warning"); return redirect(url_for("admin_dashboard"))
    return send_from_directory(REPORT_DIR, pdf_name, as_attachment=True)

@app.route("/admin/delete_report/<int:submission_id>", methods=["POST"])
@admin_required
def admin_delete_report(submission_id):
    row = db_get_submission(submission_id)
    if not row:
        flash("Not found.", "warning"); return redirect(url_for("admin_dashboard"))
    pdf_name = row[-1]
    if pdf_name:
        p = os.path.join(REPORT_DIR, pdf_name)
        try:
            if os.path.exists(p): os.remove(p)
        except Exception:
            pass
    db_update_submission_pdf(submission_id, None)
    flash("Report removed.", "info")
    return redirect(url_for("admin_dashboard"))

# Manage questions
@app.route("/admin/questions")
@admin_required
def admin_questions():
    q = load_questions()
    return render_template("admin_questions.html", questions=q)

@app.route("/admin/questions/add", methods=["GET","POST"])
@admin_required
def admin_questions_add():
    if request.method == "POST":
        category = request.form.get("category","general")
        key = request.form.get("key","").strip()
        text_en = request.form.get("text_en","").strip()
        text_hi = request.form.get("text_hi","").strip()
        text_kn = request.form.get("text_kn","").strip()
        if not key or not text_en:
            flash("Key and English text required.", "warning")
            return redirect(url_for("admin_questions_add"))
        q = load_questions()
        q.setdefault(category, []).append([key, {"en":text_en, "hi":text_hi, "kn":text_kn}])
        save_questions(q)
        flash("Added.", "success")
        return redirect(url_for("admin_questions"))
    return render_template("edit_question.html", edit=False)

@app.route("/admin/questions/edit/<category>/<int:index>", methods=["GET","POST"])
@admin_required
def admin_questions_edit(category, index):
    q = load_questions()
    try:
        item = q[category][index]
    except Exception:
        flash("Not found", "warning"); return redirect(url_for("admin_questions"))
    if request.method == "POST":
        key = request.form.get("key","").strip()
        text_en = request.form.get("text_en","").strip()
        text_hi = request.form.get("text_hi","").strip()
        text_kn = request.form.get("text_kn","").strip()
        q[category][index] = [key, {"en":text_en, "hi":text_hi, "kn":text_kn}]
        save_questions(q)
        flash("Saved", "success")
        return redirect(url_for("admin_questions"))
    return render_template("edit_question.html", edit=True, category=category, index=index, question=item)

@app.route("/admin/questions/delete/<category>/<int:index>", methods=["POST"])
@admin_required
def admin_questions_delete(category, index):
    q = load_questions()
    try:
        q[category].pop(index)
        save_questions(q)
        flash("Deleted", "info")
    except Exception:
        flash("Could not delete", "danger")
    return redirect(url_for("admin_questions"))

# serve uploads
@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname)

# ---------- Minimal fallback PDF generator used if utils not available ----------
def generate_pdf_local(data, lang="en"):
    # uses FPDF, writes into REPORT_DIR with timestamp filename, returns full path
    from fpdf import FPDF
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(REPORT_DIR, fname)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Pulmonary Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Risk Level: {data.get('risk','')}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Suggestion: {data.get('suggestion','')}")
    pdf.ln(6)
    pdf.cell(0, 8, "General Symptoms:", ln=True)
    for t in data.get("general_texts", []):
        pdf.multi_cell(0, 6, f"- {t}")
    pdf.ln(4)
    pdf.cell(0, 8, "Age-specific Symptoms:", ln=True)
    for t in data.get("age_texts", []):
        pdf.multi_cell(0, 6, f"- {t}")
    if data.get("xray_filename"):
        pdf.ln(6)
        pdf.multi_cell(0, 6, f"X-ray file: {data.get('xray_filename')}")
    pdf.output(path)
    return path

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
