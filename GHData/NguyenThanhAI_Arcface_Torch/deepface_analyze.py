import os
import argparse

from typing import List, Tuple
from itertools import groupby

from collections import Counter

import time
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn

import numpy as np
import cv2

import matplotlib.pyplot as plt

from mtcnn import MTCNN

import deepface

from deepface import DeepFace

from backbones import get_model
from utils_fn import enumerate_images


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#nums = 1000
#start = time.time()
#'''for _ in tqdm(range(nums)):
#
#    result = DeepFace.verify(img1_path=r"D:\Face_Datasets\celeba_chosen_iden\0\000023.jpg", 
#                            img2_path=r"D:\Face_Datasets\img_aligned_celeba_train_val\train\4\027099.jpg", 
#                            enforce_detection=False,
#                            model_name="ArcFace",)'''
#
#result = DeepFace.analyze(img_path=r"D:\Face_Datasets\choose_train\Antoine Griezmann\AntoineGriezmann (60).jpg",
#                          actions=["age", "gender", "race"], enforce_detection=False)
#
#end = time.time()
#print(result)
#print((end - start)/nums)


class FaceModel(nn.Module):

    def __init__(self, model_name: str="r18", num_classes: int=10177):
        super().__init__()
        self.backbone = get_model(name=model_name)

        #for layer in self.backbone.parameters():
        #    layer.requires_grad = False
        

        #in_features = self.backbone.features.out_features
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, images):
        x = self.backbone(images)
        output = self.fc(x)

        return output

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, default=r"D:\Face_Datasets\choose_train")
    parser.add_argument("--weights", type=str, default=r"D:\Face_Datasets\CelebA_Models\checkpoint.pth")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    images_dir = args.images_dir
    weights = args.weights

    images_list: List[str] = enumerate_images(images_dir=images_dir)
    images_list.sort(key=lambda x: os.path.normpath(x).split(os.sep)[-2])
    class_list = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], images_list))))
    class_list.sort()
    class_to_label = dict(zip(class_list, range(len(class_list))))
    
    class_to_images = {}

    for key, images in groupby(images_list, key=lambda x: os.path.normpath(x).split(os.sep)[-2]):
        class_to_images[key] = list(images)

    #print(class_to_images)
    print("Images list: {}".format(images_list))

    ages = {}
    gender = {}
    race = {}

    for label in tqdm(class_to_images):
        print("Label: {}".format(label))
        images = class_to_images[label]
        if label not in ages:
            ages[label] = []
        if label not in gender:
            gender[label] = []
        if label not in race:
            race[label] = []
        for image in images:
            try:
                result = DeepFace.analyze(img_path=image, actions=["age", "gender", "race"], enforce_detection=False, prog_bar=False)
                ages[label].append(result["age"])
                gender[label].append(result["gender"])
                race[label].append(result["dominant_race"])
            except Exception as e:
                print("Error: {}".format(e))
                continue

    #print("Age: {}, Gender: {}, Race: {}".format(ages, gender, race))

    ages_groups = {}
    gender_groups = {}
    race_groups = {}

    '''for group in ["0-15", "16-30", "31-45", "46-60", "60 above"]:
        ages_groups[group] = []

    for group in ["Man", "Woman"]:
        gender_groups[group] = []

    for group in ["asian", "latino hispanic", "black", "middle eastern", "indian", "white"]:
        race_groups[group] = []

    
    for person in ages:
        avg_age = np.mean(ages[person])
        if 0 <= avg_age <= 15:
            ages_groups["0-15"].append(person)
        elif 16 < avg_age <= 30:
            ages_groups["16-30"].append(person)
        elif 30 < avg_age <= 45:
            ages_groups["31-45"].append(person)
        elif 45 < avg_age <= 60:
            ages_groups["46-60"].append(person)
        else:
            ages_groups["60 above"].append(person)

    for person in gender:
        most_gender = Counter(gender[person]).most_common(1)[0][0]
        gender_groups[most_gender].append(person)

    for person in race:
        most_race = Counter(race[person]).most_common(1)[0][0]
        race_groups[most_race].append(person)'''

    for person in ages:
        avg_age = np.mean(ages[person])
        if 0 <= avg_age <= 15:
            ages_groups[person] = "0-15"
        elif 16 < avg_age <= 30:
            ages_groups[person] = "16-30"
        elif 30 < avg_age <= 45:
            ages_groups[person] = "31-45"
        elif 45 < avg_age <= 60:
            ages_groups[person] = "46-60"
        else:
            ages_groups[person] = "60 above"

    for person in gender:
        most_gender = Counter(gender[person]).most_common(1)[0][0]
        gender_groups[person] = most_gender

    for person in race:
        most_race = Counter(race[person]).most_common(1)[0][0]
        race_groups[person] = most_race

    #print("Age: {}, Gender: {}, Race: {}".format(ages_groups, gender_groups, race_groups))

    mtcnn = MTCNN()

    model = FaceModel(model_name="r50", num_classes=len(class_list))
    model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))["weights"])

    model.eval()
    model.to(device=device)

    total = {}
    total["age"] = {}
    total["gender"] = {}
    total["race"] = {}

    true_pred = {}
    true_pred["age"] = {}
    true_pred["gender"] = {}
    true_pred["race"] = {}

    accuracy = {}
    accuracy["age"] = {}
    accuracy["gender"] = {}
    accuracy["race"] = {}

    for i, image in tqdm(enumerate(images_list)):
        iden = os.path.normpath(image).split(os.sep)[-2]
        label = class_to_label[iden]
        img = Image.open(image).convert("RGB")
        #face = mtcnn.align(img=img)
        #if face is not None:
        #    img = np.array(face)
#
        #else:
        #    continue
        img = np.array(img)
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255)
        img = img.to(device=device)
        
        with torch.no_grad():
            pred = model(img)
        
        pred = pred.argmax(1).cpu().numpy()[0]
        print("Pred: {}, image: {}, iden: {}".format(pred, image, iden))

        age_iden = ages_groups[iden]
        gender_iden = gender_groups[iden]
        race_iden = race_groups[iden]

        if age_iden not in total["age"]:
            total["age"][age_iden] = 1
        else:
            total["age"][age_iden] += 1

        if gender_iden not in total["gender"]:
            total["gender"][gender_iden] = 1
        else:
            total["gender"][gender_iden] += 1

        if race_iden not in total["race"]:
            total["race"][race_iden] = 1
        else:
            total["race"][race_iden] += 1

        if label == pred:
            if age_iden not in true_pred["age"]:
                true_pred["age"][age_iden] = 1
            else:
                true_pred["age"][age_iden] += 1
            
            if gender_iden not in true_pred["gender"]:
                true_pred["gender"][gender_iden] = 1
            else:
                true_pred["gender"][gender_iden] += 1

            if race_iden not in true_pred["race"]:
                true_pred["race"][race_iden] = 1
            else:
                true_pred["race"][race_iden] += 1

    for par in total:
        for val in total[par]:
            accuracy[par][val] = true_pred[par][val] / total[par][val]

    print(accuracy)

    fig = plt.figure(figsize=(15, 5), facecolor='w')
    i = 0
    for par in accuracy:
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(par)
        ax.bar(list(accuracy[par].keys()), list(accuracy[par].values()), width=0.35)
        ax.set_ylabel("Accuracy")
        ax.set_xticklabels(list(accuracy[par].keys()))
        i += 1

    plt.show()
