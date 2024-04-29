from flask import Flask, render_template, redirect, url_for, flash, request
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from flask import jsonify
from flask_migrate import Migrate

import flask_login
from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user

from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
import random
import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from modelTraining.model_training import CNN   # this is the class for the model I made

from PIL import Image
import os
import time
# import the model here
db_name = "database.db"
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key-goes-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_name
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
migrate = Migrate(app, db)
api = Api(app)

model_name = "model3_CIFAR10"
MODEL_PATH = f"modelTraining/{model_name}.pth"
TRAINING_IMAGES_HEIGHT = 32
TRAINING_IMAGES_WIDTH = 32
TRAINING_IMAGES_CHANNELS_NUM = 3
BATCH_SIZE = 4

# NOTE: IF THESE ARE IN THE WRONG ORDER EVERYTHING IS A FAILURE, I NEED TO MAKE SURE THESE ARE IN RIGHT ORDER
# the pytorch cnn tutorial that uses cifar10 has them in this order, so I think I'm okay 
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# UPLOAD_FOLDER = '/uploaded_images'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/login', methods=["GET"])
def login_view():
    return render_template("login.html")

@app.route('/login', methods=["POST"])
def login():
    name = request.form.get('name')
    password = request.form.get('password')

    user = User.query.filter_by(name=name).first()
    if not user or not check_password_hash(user.password, password):
        flash('Invalid credentials, please try again.')
        return redirect(url_for("login_view"))
    else:
        login_user(user)
        return redirect(url_for("index"))


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route('/signup', methods=["GET"])
def signup():
    return render_template("signup.html")

@app.route('/signup', methods=["POST"])
def createUser():
    name = request.form.get('name')

    # check if user with that name exists
    usr = User.query.filter_by(name=name).first()
    if usr:
        flash('That name is taken, please use another.')
        return redirect(url_for("signup"))

    password = request.form.get('password')
    new_user = User(name=name, password=generate_password_hash(password, method='sha256'))
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for("login_view"))


# this is required for flask-login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    # maybe add code to get some info about the model here? 
    # So that I don't have to edit the html file if I decide to change the model
    return render_template("about.html")


@app.route("/ben")
def ben():
    return render_template("ben.html")


@app.route("/classifyImage", methods=['POST'])
def classify():
    img_file = request.files['image']
    img = Image.open(img_file) 
    format = img.format
    img = img.convert('RGB')  # converting to RGB sets it as 3 channel (model only works w 3 channels)
    rand = int(random.random() * 9999999)
    local_img_path = "static/submitted_images/" + str(rand) + "." + format.lower()  # rand prevents possibility of two image files w same name (would cause overwriting)
    img.save(local_img_path)

    img = Image.open(local_img_path).resize((TRAINING_IMAGES_HEIGHT, TRAINING_IMAGES_WIDTH), Image.ANTIALIAS)  # should I use RGB?
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    img = img.unsqueeze(0)  # turns from 3x32x32 into 1x3x32x32    (this accounts for the batch size dimension)

    

    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # sets dropout and batch normalization layers to evaluation mode

    prediction_tensor = model(img)
    prediction_tensor = prediction_tensor.flatten()
    index_of_max = torch.argmax(prediction_tensor)

    # put prediction through softmax layer to get percentage confidence
    sm = nn.Softmax()
    certainties = sm(prediction_tensor)
    certainty = round(round(certainties.tolist()[index_of_max], 4) * 100, 2)
    prediction = CLASSES[index_of_max]

    # create and save new prediction
    new_pred = Prediction(image_file=local_img_path, prediction=prediction, certainty=certainty, user_id=current_user.id)
    db.session.add(new_pred)
    db.session.commit()

    return render_template("prediction.html", prediction=prediction, certainty=certainty)


@app.route("/predictions")
def predictions():
    return render_template("predictions.html", all_users=True)

@app.route("/yourPredictions")
def your_predictions():
    return render_template("predictions.html", all_users=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_file = db.Column(db.String(100), nullable=False)
    date_submitted = db.Column(db.String(100), nullable=False, default=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    prediction = db.Column(db.String(100), nullable=False)
    certainty = db.Column(db.Integer, default=0)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))


    def serialize(self): 
        return {
            "id": self.id,
            "image_filepath": self.image_file,
            "date_submitted": self.date_submitted,
            "prediction": self.prediction,
            "certainty": self.certainty,
            "user": User.query.filter_by(id=self.user_id).first().name
        }

class User(db.Model, UserMixin):  # the UserMixin class provides implementations of is_authenticated(), is_active(), is_anonymous(), and get_id()
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000), unique=True)
    password = db.Column(db.String(100))   # how do I hash what the user submits?
    # need to make a one to many relationship with predictions
    predictions = db.relationship('Prediction', backref='user')  # backref serves the same purpose as Django's related_name

    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,
            "predictions": list(self.predictions)  # TODO I really feel like this isn't the right way to do this
        }


db.create_all()  # don't think I need this line ?


class GetPredictions(Resource):
    def get(self):
        serialized_predictions = [prediction.serialize() for prediction in Prediction.query.all()]
        return serialized_predictions

class GetPredictionsByUser(Resource):
    def get(self):
        serialized_predictions = [prediction.serialize() for prediction in Prediction.query.filter_by(user_id=current_user.id).all()]
        return serialized_predictions

class GetUsername(Resource):
    def get(self, id):
        user = User.query.filter_by(id=id).first()
        return [ user.name ]

api.add_resource(GetPredictions, "/getPredictions")
api.add_resource(GetPredictionsByUser, "/getPredictionsByUser") 
api.add_resource(GetUsername, "/getUsername/<id>")
db.create_all()

if __name__ == "__main__":
    app.run(debug=True)