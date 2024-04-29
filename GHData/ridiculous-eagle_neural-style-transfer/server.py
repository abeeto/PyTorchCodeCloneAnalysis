from gevent import pywsgi
import gevent, app, os
import logging
import log_config


log_config.load_settings()
logger = logging.getLogger(__name__)
try:
    logger.info("Start Serving at 21050")
    http_server = pywsgi.WSGIServer(('0.0.0.0', 21050), app.flask_app)
    http_server.serve_forever()
except Exception as ex:
    logger.error(ex)
    exit(-1)
