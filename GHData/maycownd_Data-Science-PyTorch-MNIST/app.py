import os
import string
import random
import json
import requests
import numpy as np
import sys
sys.path.append('Code/')
from Code.preprocessing.fmnist import output_label
from Code.training.pytorch_train import load_model
import io
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template, jsonify, request
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

"""
Constants
"""
MODEL_URI = 'http://localhost:5000/v1/models/fmnist:predict'
OUTPUT_DIR = 'static/'
SIZE = 28

"""
Utility functions
"""


def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
                             [0.0],
                             [1.0])
        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes,
                   model_path="Data/models/ConvNet_loss_0.09.pth"):
    model = load_model(model_path, eval=True)
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)
    with torch.no_grad():
        outputs = model.forward(tensor)
        outputs = F.softmax(outputs)
        _, y_hat = outputs.max(1)
    return output_label(y_hat), y_hat

# define the function to get the class predicted of image
# it takes the parameter: image path and provide the output as the predicted class


def get_category(image_path,
                 model_path="Data/models/ConvNet_loss_0.09.pth"):
    # read the image in binary form
    model = load_model(model_path)
    model.eval()
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    # transform the image
    transformed_image = transform_image(image_bytes=image_bytes)
    # use the model to predict the class
    outputs = model.forward(transformed_image)
    _, category = outputs.max(1)
    # return the value
    return output_label(category)


"""
Routes
"""
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name, _ = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                file_name = generate_filename()
                image_path = os.path.join(OUTPUT_DIR, file_name)
                uploaded_file.save(image_path)
                class_name = get_category(image_path)
                directory = os.getcwd()
                result = {
                    'class_name': class_name,
                    'path_to_image': image_path,
                    'size': SIZE,
                }
                return render_template('show.html', result=result)
    return render_template('index.html')


if __name__ == '__main__':
    # with open("Data/tests/yellow-shirt.png", 'rb') as f:
    #     image_bytes = f.read()
    #     print(get_prediction(image_bytes=image_bytes))
        
    # with open("Data/tests/ankle boot.png", 'rb') as f:
    #     image_bytes = f.read()
    #     print(get_prediction(image_bytes=image_bytes))
        
    # with open("Data/tests/pullover.png", 'rb') as f:
    #     image_bytes = f.read()
    #     print(get_prediction(image_bytes=image_bytes))
        
    # with open("Data/tests/sneaker.png", 'rb') as f:
    #     image_bytes = f.read()
    #     print(get_prediction(image_bytes=image_bytes))
        
    # with open("Data/tests/trouser.png", 'rb') as f:
    #     image_bytes = f.read()
    #     print(get_prediction(image_bytes=image_bytes))
    app.run(debug=True)