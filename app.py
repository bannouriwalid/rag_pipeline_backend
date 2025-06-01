from flask import Flask
from flask_cors import CORS
from config import Config
from db import db
from api.user_routes import user_bp


def create_app():
    application = Flask(__name__)
    application.config.from_object(Config)
    db.init_app(application)
    CORS(application)
    application.register_blueprint(user_bp, url_prefix="/api/user")
    return application


app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
