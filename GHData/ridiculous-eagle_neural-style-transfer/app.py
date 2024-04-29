import os
import cv2
import imutils
import urllib
import flask
import logging
import traceback
import image_proc


logger = logging.getLogger(__name__)
flask_app = flask.Flask(__name__)


@flask_app.route('/')
def do():
    try:
        image_url = urllib.parse.unquote(flask.request.args.get('image'))
        model_path = f'models/{urllib.parse.unquote(flask.request.args.get("model"))}.t7'
        canvas_url = flask.request.args.get('canvas')
        if canvas_url is None:
            canvas_url = 'backgrounds/white.jpg'
        else:
            canvas_url = urllib.parse.unquote(canvas_url)
        canvas_width = flask.request.args.get('width')
        try:
            canvas_width = int(canvas_width)
        except:
            canvas_width = 800
        overlay_offset = flask.request.args.get('offset')
        try:
            overlay_offset = int(overlay_offset)
        except:
            overlay_offset = 100
    except Exception as ex:
        logger.error(ex)
        return f'parameter exception raised: {ex}', 400
    if os.path.exists(model_path):
        try:
            net = cv2.dnn.readNetFromTorch(model_path)
            canvas = image_proc.get_image(canvas_url)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2RGBA)
            canvas = imutils.resize(canvas, width = canvas_width)
            (height, width) = canvas.shape[:2]
            logger.info(f"canvas uri: { canvas_url }, width: { width }, height: { height }")
            overlay = image_proc.get_image(image_url)
            overlay = image_proc.crop(overlay, 25)
            overlay = imutils.resize(overlay, width = canvas_width - overlay_offset * 2)
            (image_height, image_width) = overlay.shape[:2]
            if image_height > height:
                overlay = imutils.resize(overlay, height = height)
                (image_height, image_width) = overlay.shape[:2]
            logger.info(f"overlay uri: { image_url }, width: { image_width}, height: { image_height }")
            image_proc.overlay_images(canvas, overlay, int((width - image_width) / 2), int((height - image_height) / 2))
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2RGB)
            image_output = image_proc.style_transfer(canvas, net, (103.939, 116.779, 123.680))
            retval, buffer = cv2.imencode('.jpg', image_output)
            response = flask.make_response(buffer.tobytes())
            response.headers['Content-Type'] = 'image/jpg'
            return response
        except Exception as ex:
            logger.error(ex, traceback.format_exc())
            return f'exception raised: {ex}', 500
    else:
        logger.error(ex)
        return f'model not found: {model_path}', 404
