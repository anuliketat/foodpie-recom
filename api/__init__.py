import os
from importlib import import_module

from flask import Flask, g, jsonify, request
from flask_pymongo import PyMongo
from gridfs import GridFS


def register_blueprints(app):
    current_directory = os.path.abspath(os.path.dirname(__file__))
    modules = [name for name in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, name)) and name != '__pycache__']
    for name in modules:
        try:
            mod = import_module('.{}'.format(name), 'api')
            app.register_blueprint(mod.blue_print)
        except ModuleNotFoundError:
            error_message = name + " module is missing or it lacks a blueprint."
            raise Exception(error_message)
    return None

def register_middleware(app, mongo_main, mongo_ai):
    @app.before_request
    def before_request():
        g.db_main = mongo_main.db
        g.db_ai = mongo_ai.db

        g.fs_ai = GridFS(mongo_ai.db)

    # ERROR HANDLER
    @app.errorhandler(404)
    def not_found(error):
        return "Page not found!", 404

def create_app(testing=False):
    app = Flask("foodpie")
    app.config.from_object('config')

    # ESTABLISH & INJECT DB CONNECTION
    kwargs = {}
    kwargs['tz_aware'] = True
    if app.config['MONGO_AUTH']:
        kwargs['username'] = app.config['MONGO_USERNAME']
        kwargs['password'] = app.config['MONGO_PASSWORD']
        kwargs['authSource'] = app.config['MONGO_AUTH_SOURCE']

    mongo_main = PyMongo(app, uri=app.config["MONGO_URI_MAIN"], **kwargs)
    mongo_ai = PyMongo(app, uri=app.config["MONGO_URI_AI"], **kwargs)

    register_blueprints(app)
    register_middleware(app, mongo_main, mongo_ai)

    # Home location. Will give the welcome address for testing the API
    @app.route('/')
    def index():
        return jsonify({
            "message": ("Welcome to FOODPIE "
                        + app.config['ECOSYSTEM']
                        + " API")
            })

    return app

app = create_app()
