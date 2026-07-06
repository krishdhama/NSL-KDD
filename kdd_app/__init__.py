import os

from flask import Flask

from .routes import web


def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")
    app.register_blueprint(web)
    return app
