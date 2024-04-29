import base64 #decoding camera images
import numpy as np #matrix math
import socketio #real-time server
import eventlet #concurrent networking
import eventlet.wsgi #web server gateway interface
from PIL import Image #image manipulation
from flask import Flask #web framework
from io import BytesIO #input output
import additional_files.utils #helper class


###############################################################################################
# This is reused code from rep used for comunication between unity simlator and our CNN
# https://github.com/llSourcell/How_to_simulate_a_self_driving_car
###############################################################################################

import model_architecture_color
import torch
import cv2

# Seach for cuda card and if available activate
print("Cuda is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the cpu")

# Create and load a model
model = (model_architecture_color.CNN()).to(device)
# Type in your path to the model you want to load. Watch out for the architecture and color/no color
model.load_state_dict(torch.load("track_2_model_color.pt"))
model.eval()


MAX_SPEED = 18
#################################################################

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init image array as empty
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle,throttle,speed
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])

        try:
            # Center image of the car
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            image = np.asarray(image)       # from PIL image to numpy array
            image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # To grascale from RGB
            image_color = cv2.resize(image_color,(80,40)) # Resize in order to be able to input to CNN
            steering_angle = 0.0

            # Convert to tensor and normalize 0.0-1.0
            input = torch.tensor(image_color,  dtype=torch.float).view(1,3,40,80).to(device)/255.0
            speed_input = torch.tensor(speed, dtype=torch.float).view(1,1).to(device)/30
            # Uncomment for feed forward neural network input
            #input = torch.tensor(image_gray,  dtype=torch.float).view(1,1,40*80).to(device)/255.0

            # Netowrk prediction
            prediction = model(input, speed_input)

            # Get float from prediction
            steering_angle = prediction[0][0][0].item()

            # Set max speed and set throttle to reverse if our speed is higher
            if speed < MAX_SPEED:
                throttle = prediction[0][0][1].item()
            else:
                throttle = -0.2

            print("Predicted:", prediction)

            # Show the input for the CNN
            image_plot = cv2.resize(image_color,(640,320))
            cv2.imshow('image',image_plot)
            cv2.waitKey(1)

            # Send controlls to the car
            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

# wrap Flask application with engineio's middleware
app = socketio.Middleware(sio, app)
# deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
