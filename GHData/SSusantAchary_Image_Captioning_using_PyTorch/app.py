
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
@author:Susant_Achary
'''

import torch
import pickle 
import os

'''import the dependency module'''
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image, ImageDraw, ImageFont

'''import  the flask module, an object of flask is our WSGI application'''
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

'''flask constructor'''
app = Flask(__name__)

'''
define the location of upload image folder location
create a SET for required format of image to be processed only.
'''
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

'''check for the device capable of cpu or cuda enabled hardware'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''validates for input image format in the defined set'''
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

'''
Device Configuration
route()- a type of Flask class decorator, tells application which associated url to be called
'''
@app.route('/')
def index():
    return render_template('index.html')

'''data transfer through POST protocol'''
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        return redirect(url_for('uploaded_file',filename=filename))


'''loaded image needs to be resized/baseline image resolution to allows the CNN to process '''
def load_image(image_path, transform=None):
    print("!!!Inside_image_path",image_path)
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


'''route to the result image location'''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print("####Entry File Name",filename)
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    print("*******PRINT******", TEST_IMAGE_PATHS)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    '''Load vocabulary wrapper'''
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    '''created instance to build models'''
    encoder = EncoderCNN(256).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(256, 512, len(vocab), 1)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    '''
    Load the trained model parameters 
    EncoderCNN pickle- objects detection
    Decoder RNN pickle pretrained- sequence prediction
    '''
    encoder.load_state_dict(torch.load('models/encoder-5-3000.pkl'))
    decoder.load_state_dict(torch.load('models/decoder-5-3000.pkl'))


    for img in TEST_IMAGE_PATHS:
        '''Prepare an image'''
        image = load_image(img, transform)
        image_tensor = image.to(device)

        '''Generate an caption from the image'''
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()

        '''Convert word_ids to words'''
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<start>' :#or word == '<end>':# or word == '.':
                continue
            if word == '<end>':
                break
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)
        '''
        Print the sentence in the console
        read the image and overlay the predicted text on the image
        save the result image
        route/return the saved image result location as output
        '''
        print (sentence)
        print("FileName",img)
        image = Image.open(img)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size=15)
        (x,y) = (10,10)
        color = 'rgb(244,208,63)'
        draw.text((x,y),sentence,fill=color,font=font)
        image.save('uploads/'+filename)

    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
    
if __name__ == '__main__':
    '''
    run() method starts/triggers the application on local/VM instance server
    host: is the IP to be run and accessible
    port: 5000 for local/ 80 for cloud(as used in GCP for HTTP request over internet)
    threaded: True to enable to access multiple request
    '''
    app.run(debug = True, threaded = True, host ='0.0.0.0',port=5000)
