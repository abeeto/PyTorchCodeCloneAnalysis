import os
from typing import List
from fastapi import FastAPI, UploadFile, File

import numpy as np
import cv2
import torch

import uvicorn

from backbones import get_model
from facenet_pytorch import InceptionResnetV1

import vggface2_models.resnet as ResNet
import vggface2_models.senet as SENet
from vggface2_models.utils import load_state_dict

from mtcnn import MTCNN

from face_database import FaceRecognitionDataBase

from utils_fn import read_image, compare_faces

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = "r50"
weights = r"C:\Users\Thanh\Downloads\backbone.pth"
db_dir = "db"
db_name = "face_db.db"
model_type = "arcface"

db_file = os.path.join(db_dir, "_".join([model_type, db_name]))

if model_type == "arcface":
    model = get_model(name=network)
    model.load_state_dict(torch.load(weights))
elif model_type == "facenet":
    model = InceptionResnetV1(pretrained="vggface2")
elif model_type == "vggface2":
    model = ResNet.resnet50(num_classes=8631, include_top=False)
    load_state_dict(model, weights)
else:
    raise ValueError("Unnsupported model type {}".format(model_type))
model.eval()
model.to(device)

mtcnn = MTCNN()
db = FaceRecognitionDataBase(db_file)


def extract_feature(file: bytes) -> np.ndarray:
    image = read_image(file)
    face = mtcnn.align(img=image)
    if face is None:
        return face

    img = np.array(face)
    if model_type == "vggface2":
        img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    if model_type == "vggface2":
        img = img.astype(np.float32)
        img = img - np.array([91.4953, 103.8827, 131.0912])[:, np.newaxis, np.newaxis]
        img = torch.from_numpy(img).unsqueeze(0).float()
    else:
        img = torch.from_numpy(img).unsqueeze(0).float()
        #print(img.shape)
        img.div_(255).sub_(0.5).div_(0.5)
        
    img = img.to(device=device)
    with torch.no_grad():
        feature = model(img).cpu().numpy()[0]

    feature = feature / np.linalg.norm(feature)

    return feature

@app.post("/register")
def register_face(info: str, files: List[bytes]=File(...)):
    features = []
    print("Number of files: {}".format(len(files)))
    for file in files:
        '''image = read_image(file)
        face = mtcnn.align(img=image)
        if face is None:
            continue

        img = np.array(face)
        if model_type == "vggface2":
            img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        if model_type == "vggface2":
            img = img.astype(np.float32)
            img = img - np.array([91.4953, 103.8827, 131.0912])[:, np.newaxis, np.newaxis]
            img = torch.from_numpy(img).unsqueeze(0).float()
        else:
            img = torch.from_numpy(img).unsqueeze(0).float()
            #print(img.shape)
            img.div_(255).sub_(0.5).div_(0.5)
        
        img = img.to(device=device)
        with torch.no_grad():
            feature = model(img).cpu().numpy()[0]

        if len(feature.shape) > 1:
            feature = np.squeeze(feature)
        feature = feature / np.linalg.norm(feature)'''
        feature = extract_feature(file=file)
        if feature is None:
            continue
        features.append(feature)

        if len(features) > 0:
            print("Number of features: {}".format(len(features)))
            features = np.stack(features, axis=0)
            print("Number of features: {}".format(features.shape[0]))
            db.add_persons(info=info, features=features)
            last_id = db.get_latest_id()[0]
            return {"id": last_id, "info": info}

        else:
            return {"No face to add"}


@app.post("/verify/{person_id}")
def verify_face(person_id: int, threshold: float, image: bytes=File(...)):
    feature = extract_feature(file=image)
    if feature is None:
        return {"No face"}

    persons = db.get_person_by_id(person_id=person_id)
    print("Len persons: {}".format(len(persons)))
    if len(persons) > 0:
        assert len(persons) == 1
        person = persons[0]
        person_info = person[1]
        person_feature = person[2]
        response = compare_faces(features_1=person_feature, feature_2=feature, threshold=threshold)
        result = list(map(lambda x: bool(x), result))
        return {"result": result, "info": person_info}
    else:
        return {"No person id {}".format(person_id)}


@app.delete("/delete/{person_id}")
def delete_id(person_id: int):
    db.delete_by_id(person_id=person_id)
    return {"id": person_id}


@app.get("/latest_id")
def get_latest_id():
    person = db.get_latest_id()
    person_id = person[0]
    person_info = person[1]
    return {"id": person_id, "info": person_info}
