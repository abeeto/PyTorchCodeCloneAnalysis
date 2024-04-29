from flask import Flask, send_from_directory
from jinja2 import Template,Environment,FileSystemLoader
import torch as th
from model import CNN

app= Flask(__name__,static_folder="static") # create an app for the website
file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)
t = env.get_template('index.html')
d = range(20)
m = CNN()
m.load_state_dict(th.load('cnn.pt'))
m.eval()
dataset = th.load("face_dataset.pt")# load face image dataset
X =  dataset["X"]

@app.route("/")
def result():
    return t.render(face_ids=d)

@app.route("/predict/<int:face_id>")
def predict(face_id):
    if face_id%2 == 0:
        i = face_id //2
    else:
        i = face_id //2 + 10
    x = X[i].reshape(1,1,64,64)
    z=m(x)
    if z[0,0]>0:
        p = 'Owner' 
    else:
        p = 'Not Owner'
    return t.render(face_ids=[face_id], p=p)

if __name__ == "__main__":
    app.run()
