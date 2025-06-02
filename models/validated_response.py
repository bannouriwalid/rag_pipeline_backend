from db import db
from datetime import datetime


class ValidatedResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    response_id = db.Column(db.Integer, db.ForeignKey('response.id'), nullable=False, unique=True)
    confirmed = db.Column(db.Boolean, nullable=False)
    correction = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
