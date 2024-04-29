import numpy as np
import cv2
import pafy
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

import torch
from torch import nn
from torchvision import transforms


url = "https://www.youtube.com/watch?v=wqctLW0Hb_0"

def get_youtube(url):
    play = pafy.new(url).streams[-1] # we will take the lowest quality stream
    assert play is not None # makes sure we get an error if the video failed to load
    return cv2.VideoCapture(play.url)

def transform(img):
    tsfm =  transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
    return tsfm(img)

def crop_img(img):
        if len(img.shape) == 3:
            y = img.shape[0]
            x = img.shape[1]
        elif len(img.shape) == 4:
            y = img.shape[1]
            x = img.shape[2]
        else:
            raise ValueError(f"Image shape: {img.shape} invalid")

        out_size = min((y, x))
        startx = x // 2 - out_size // 2
        starty = y // 2 - out_size // 2

        if len(img.shape) == 3:
            return img[starty:starty+out_size, startx:startx+out_size]
        elif len(img.shape) == 4:
            return img[:, starty:starty+out_size, startx:startx+out_size]

def plot_boxes(output_img, labels, boxes):
    for label, (x1, y1, x2, y2) in zip(labels, boxes):
        if (x2 - x1) * (y2 - y1) < 0.25:
            x1 = int(x1*output_img.shape[1])
            y1 = int(y1*output_img.shape[0])
            x2 = int(x2*output_img.shape[1])
            y2 = int(y2*output_img.shape[0])
            rgba = cmap(label)
            bgr = rgba[2]*255, rgba[1]*255, rgba[0]*255
            cv2.rectangle(output_img, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(output_img, classes_to_labels[label - 1], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return output_img[:,:,::-1]


cap = get_youtube(url)
ret, frame = cap.read()
print(type(frame))
plt.imshow(frame[:,:,::-1]) # OpenCV uses BGR, whereas matplotlib uses RGB
plt.show()

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').eval().to(torch.device("cpu"))
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

threshold = 0.2
cmap = cm.get_cmap("tab10_r")
device = torch.device("cpu")
classes_to_labels = utils.get_coco_object_dictionary()

img_tens = transform(Image.fromarray(frame[:,:,::-1])).unsqueeze(0).to(device)

results = utils.decode_results(model(img_tens))
boxes, labels, conf = utils.pick_best(results[0], threshold)
output_img = crop_img(frame)

img_box = plot_boxes(output_img, labels, boxes)

plt.imshow(img_box)
plt.show()
cap.release()

