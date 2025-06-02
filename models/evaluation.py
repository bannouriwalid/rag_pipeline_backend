from db import db
from datetime import datetime


class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    context = db.Column(db.Text)
    response = db.Column(db.Text)
    label = db.Column(db.Text)
    clarity_score = db.Column(db.Float)
    exactitude_score = db.Column(db.Float)
    context_adherence_score = db.Column(db.Float)
    relevance_score = db.Column(db.Float)
    completeness_score = db.Column(db.Float)
    logical_flow_score = db.Column(db.Float)
    uncertainty_handling_score = db.Column(db.Float)
    overall_feedback = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

