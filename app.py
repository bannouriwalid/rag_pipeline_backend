from flask import Flask
from flask_cors import CORS
from config import Config
from db import db
from api.user_routes import user_bp
from api.admin_routes import admin_bp
from api.doctor_routes import doctor_bp


def create_app():
    application = Flask(__name__)
    application.config.from_object(Config)
    db.init_app(application)
    CORS(application)
    application.register_blueprint(user_bp, url_prefix="/api/user")
    application.register_blueprint(admin_bp, url_prefix="/api/admin")
    application.register_blueprint(doctor_bp, url_prefix="/api/doctor")
    return application


app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
