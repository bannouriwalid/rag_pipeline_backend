from db import db
from datetime import datetime


class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)
    generated_answer = db.Column(db.Text)
    status = db.Column(db.String(50), default="pending")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
