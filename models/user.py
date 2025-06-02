from db import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    role = db.Column(db.Enum('admin', 'doctor', 'patient', name='user_roles'), default='patient', nullable=False)

