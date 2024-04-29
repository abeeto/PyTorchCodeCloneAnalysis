# -*- coding: utf-8 -*-
# usage: python3 classificator.py
# request to do: POST http://localhost:8080/classify

from   PIL import Image
from   io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.optim import lr_scheduler
import torchvision
from   torchvision import datasets, models, transforms
import torchvision.transforms as T
import cv2
import numpy as np
import base64
import os
import json

from http.server import BaseHTTPRequestHandler, HTTPServer

serverPort = 8080
label_path = "./labels_27_03_2020.txt"
model_path = "./model_classification_vertical_27_03_2020.pt"

device = torch.device("cpu")

# Preprocess images while loading them
loader = transforms.Compose([
    transforms.Resize(229),  # scale imported image
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # transform it into a torch tensor

# Resize an image to a given square size while preserving width and height ratio
def make_square(image, square_size):
    height = np.size(image, 0)
    width  = np.size(image, 1)
    if(height > width):
        differ = height
    else:
        differ = width
    differ += 4
    mask  = np.zeros((differ, differ, 3), dtype="uint8")
    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)
    mask[y_pos:y_pos + height, x_pos:x_pos + width] = image[0:height, 0:width]
    mask = cv2.resize(mask, (square_size, square_size),
                      interpolation=cv2.INTER_CUBIC)
    return mask

# Load image from disk
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Load the model on the cpu
def load_model():
    labels = []
    with open(label_path, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            labels.append(currentPlace)
    device      = torch.device("cpu")
    model_ft    = models.resnext101_32x8d(pretrained=False)
    num_ftrs    = model_ft.fc.in_features
    nb_classes  = len(labels)
    model_ft.fc = nn.Linear(num_ftrs, nb_classes)
    model_ft    = model_ft.to(device)
    model_ft.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device('cpu')))
    model_ft.eval()

    to_return = list()
    to_return.append(model_ft)
    to_return.append(labels)
    return to_return

# Inference
def classify_img(img_bin, model):
    jpg_as_np = np.frombuffer(img_bin, dtype=np.uint8)
    img       = cv2.imdecode(jpg_as_np, flags=1)
    img       = make_square(img, 229)
    # TODO: stop writing/reading image from the disk
    cv2.imwrite("/tmp/img.jpg", img)
    img = image_loader("/tmp/img.jpg")
    os.remove("/tmp/img.jpg")

    prediction     = model(img)
    ps             = torch.nn.functional.softmax(prediction)
    topk, topclass = ps.topk(1, dim=1)
    label          = labels_py[topclass.cpu().numpy()[0][0]]
    class_img      = label
    return label

# Convert image to base64
def img_from_base64(base64_data):
    base64_img_bytes   = base64_data
    decoded_image_data = base64.b64decode(base64_img_bytes)
    return decoded_image_data

# Main function
def doClassification(self):
    content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
    img_data       = self.rfile.read(content_length)     # <--- Gets the data itself

    print("POST request,\nPath: %s\nHeaders:\n%s\n\n" %
          (str(self.path), str(self.headers)))

    img         = img_from_base64(img_data)
    class_label = classify_img(img, model)
    label_split = class_label.split('.')

    self.send_response(200)
    self.send_header('Content-type', 'application/json')
    self.end_headers()

    response = {}
    response['elementType']     = 'SIGN'
    response['type']            = label_split[4]
    response['elementCategory'] = label_split[3]

    self.wfile.write(json.dumps(response).encode())

# HTTPServer class
class myHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/classify':
            doClassification(self)

# Model loading
AI_info   = load_model()
model     = AI_info[0]
labels_py = AI_info[1]

# Start the web service
server = HTTPServer(('0.0.0.0', serverPort), myHandler)
print('Started httpserver on port ', serverPort)
# Wait forever for incoming http requests
server.serve_forever()
