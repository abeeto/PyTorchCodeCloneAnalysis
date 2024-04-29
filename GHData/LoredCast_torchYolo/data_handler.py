import cv2
import os
import xml.etree.cElementTree as ET
import numpy as np
from tqdm import tqdm
import pickle


IMAGES_PATH = r"./data/Images/"
ANNONATIONS_PATH = r"./data/annotations/"

TRAINING_FILE = "training_sample"
TESTING_FILE = "test_sample"

N_SAMPLES = len(os.listdir(IMAGES_PATH))

TRAINING = 0.8

N_TRAINING = int(TRAINING * N_SAMPLES)


training_data = []
testing_data = []



def parseXML(file):
    tree = ET.ElementTree(file=file)
    root = tree.getroot()
    data = {
        "faces": []
    }

    size_ = root.find("size")
    w = int(size_.find("width").text)
    h = int(size_.find("height").text)

    data["dims"] = (w, h)


    for element in root.findall("object"):
        box = element.find("bndbox")
        x1 = int(box.find("xmin").text)
        y1 = int(box.find("ymin").text)
        x2 = int(box.find("xmax").text)
        y2 = int(box.find("ymax").text)

        mask_ = element.find("name").text

        mask = False if mask_ == "without_mask" else True

        data["faces"].append({
            "box": (x1, y1, x2, y2),
            "mask": mask
            })

    return data

def makeTrainingData():
    print("Loading Training Data...")
    for f in tqdm(os.listdir(IMAGES_PATH)[:N_TRAINING]):
        try:
            path = os.path.join(IMAGES_PATH, f)
            im = cv2.imread(path)
            tag = f.strip(".png") + ".xml"
            data = parseXML(os.path.join(ANNONATIONS_PATH, tag))
            training_data.append([np.array(im), data])
        except:
            pass
    print("Done!")

def makeTestingData():
    print("Loading Testing Data...")
    for f in tqdm(os.listdir(IMAGES_PATH)[N_TRAINING:]):
        try:
            path = os.path.join(IMAGES_PATH, f)
            im = cv2.imread(path)
            tag = f.strip(".png") + ".xml"
            data = parseXML(os.path.join(ANNONATIONS_PATH, tag))
            testing_data.append([np.array(im), data])
        except:
            pass
    print("Done!")

if __name__ == "__main__":
    makeTrainingData()
    makeTestingData()
    training_file = open(TRAINING_FILE, "wb")
    testing_file = open(TESTING_FILE, "wb")

    pickle.dump(training_data, training_file)
    pickle.dump(testing_data, testing_file)

    training_file.close()
    testing_file.close()
