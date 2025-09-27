from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    surveys = db.relationship('SurveyResult', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SurveyResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    answers = db.Column(db.Text)  # store as JSON
    age_based = db.Column(db.Text)  # JSON too
    score = db.Column(db.Float)
    risk = db.Column(db.String(20))
    xray_required = db.Column(db.Boolean, default=False)
    xray_uploaded = db.Column(db.String(120))
    xray_prediction = db.Column(db.String(120))
    pdf_report = db.Column(db.String(120))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def set_answers(self, answers_dict):
        self.answers = json.dumps(answers_dict)

    def get_answers(self):
        return json.loads(self.answers) if self.answers else {}

    def set_age_based(self, age_dict):
        self.age_based = json.dumps(age_dict)

    def get_age_based(self):
        return json.loads(self.age_based) if self.age_based else {}
