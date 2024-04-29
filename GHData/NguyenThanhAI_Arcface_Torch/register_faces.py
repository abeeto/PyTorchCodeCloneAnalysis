import os
import argparse
from typing import List, Dict
from itertools import groupby

from PIL import Image

import numpy as np
import cv2
import torch
from yaml import parse

from backbones import get_model
from facenet_pytorch import InceptionResnetV1

import vggface2_models.resnet as ResNet
import vggface2_models.senet as SENet
from vggface2_models.utils import load_state_dict

from mtcnn import MTCNN
from utils_fn import enumerate_images
from face_database import FaceRecognitionDataBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(100)

def get_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\hand_faces")
    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    parser.add_argument("--num_images_per_id", type=int, default=5)
    parser.add_argument("--db_file", type=str, default=r"db\face_db.db")
    parser.add_argument("--network", type=str, default="r50")
    #parser.add_argument("--weights", type=str, default=r"C:\Users\Thanh\Downloads\backbone.pth")
    parser.add_argument("--weights", type=str, default=r"C:\Users\Thanh\Downloads\resnet50_ft_weight.pkl")
    #parser.add_argument("--model_type", type=str, default="facenet")
    parser.add_argument("--model_type", type=str, default="vggface2")

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()

    images_dir = args.images_dir
    num_images_per_id = args.num_images_per_id
    #db_file = args.db_file
    

    network = args.network
    weights = args.weights
    model_type = args.model_type

    db_dir = os.path.dirname(os.path.normpath(args.db_file))
    db_name = os.path.basename(args.db_file)

    db_file = os.path.join(db_dir, "_".join([model_type, db_name]))
    print(db_file)
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

    images_list: List[str] = enumerate_images(images_dir=images_dir)

    #images_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
    images_list.sort()

    id_to_images: Dict[str, List[str]] = {}
    for keys, items in groupby(images_list, key=lambda x: x.split(os.sep)[-2]):
        id_to_images[keys] = list(items)
    

    for identity in id_to_images:

        #images = np.random.choice(id_to_images[identity], size=num_images_per_id, replace=False).tolist()
        images = id_to_images[identity][:num_images_per_id]
        features = []

        for image in images:
            img = Image.open(image).convert("RGB")
            face = mtcnn.align(img=img)
            if face is None:
                continue
            img = np.array(face)
            img = np.array(img)
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
            #print(img.shape)

            with torch.no_grad():
                feature = model(img).cpu().numpy()[0]
            
            if len(feature.shape) > 1:
                feature = np.squeeze(feature)
            feature = feature / np.linalg.norm(feature)
            print(feature.shape)
            features.append(feature)

        if len(features) > 0:
            print("Number of features: {}".format(len(features)))
            features = np.stack(features, axis=0)
            print("Number of features: {}".format(features.shape[0]))
            db.add_persons(info=identity, features=features)
            last_id = db.get_latest_id()[0]
            print("id: {}, info: {}".format(last_id, identity))

        else:
            print("No face to add")
