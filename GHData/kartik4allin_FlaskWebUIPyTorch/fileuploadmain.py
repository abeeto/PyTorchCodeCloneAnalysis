from __future__ import print_function
import os
#import magic
import urllib.request
import torch
from flask import Flask, flash, request, redirect, render_template,jsonify
from werkzeug.utils import secure_filename
from flaskwebgui import FlaskUI #get the FlaskUI class
import io
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import json



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static')
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# Feed it the flask app instance 
ui = FlaskUI(app)


imagenet_class_index = json.load(open('C:\\AI_Project\\FlaskWeUIExample\\imagenet_class_index.json'))

    
# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()    

    
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        print('121212')
        files = request.files.getlist('files[]')
        for file in files:
            # convert that to bytes
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('3434'+filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print('45545'+full_filename)
            with open(full_filename, 'rb') as f:
                image_bytes = f.read()
            #img_bytes = full_filename.read()
            print('67677'+full_filename)
            class_id, class_name = get_prediction(image_bytes=image_bytes)
            print('78787'+class_name)
            print('89898989'+full_filename)            
        flash('File(s) successfully uploaded')
    return render_template('upload.html',value=class_name,user_image = full_filename)
    

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]



# call the 'run' method
ui.run()
