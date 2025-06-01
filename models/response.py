from db import db
from datetime import datetime


class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    question = db.Column(db.Text)
    generated_answer = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)