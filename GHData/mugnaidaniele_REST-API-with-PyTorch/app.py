from flask import Flask, jsonify, request
from utils import get_prediction_idx, get_model
import json

app = Flask(__name__)
classes = json.load(open("utils/imagenet_class_index.json"))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is None or file.filename == "":
            return jsonify({'error': "no file"})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        try:
            img_bytes = file.read()
            model = get_model()
            idx = get_prediction_idx(img_bytes, model)
            prediction = classes[idx][1]
        except:
            return jsonify({'error': 'error during inference'})
    return jsonify({'category': prediction})
