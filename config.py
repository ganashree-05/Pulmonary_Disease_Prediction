import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Secret key (for sessions, forms)
SECRET_KEY = "supersecretkey123"

# Database path
DB_PATH = os.path.join(BASE_DIR, "pulmo.db")

# Upload directories
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(BASE_DIR, "model", "pneumonia_model.h5")

# Dataset folder (for training)
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")

# Flask-SQLAlchemy config
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Ensure folders exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
