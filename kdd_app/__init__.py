import os

from flask import Flask

from .routes import web


def create_app():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_dir = os.path.join(base_dir, "templates")

    app = Flask(__name__, template_folder=template_dir)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")
    app.register_blueprint(web)
    return app
