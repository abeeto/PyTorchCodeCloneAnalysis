import log_config
from app import flask_app


log_config.load_settings()
application = flask_app

if __name__ == '__main__':
    application.run()
