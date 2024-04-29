# -*- coding: utf-8 -*-
"""
=================================================================
Title: test.py
Syntax: python test.py
Description: Test the saved NN model to detect the gestures over a real-time video stream that acquires frames. Press
"a" to stop the gesture recognition process
=================================================================
"""

__author__ = "Aparajit Balaji"

# Importing necessary modules

import cv2
import torch
from PIL import Image
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

gesture_names = ["none", "play", "stop", "vol-down", "vol-up"]

# Loading the stored model

dl_model = torch.load("gesture_model.pth")
dl_model.eval()

# Transformations to be applied over the image

img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# Opening the webcam

cap = cv2.VideoCapture(0)

# Code to read the frames

if not cap.isOpened():
    print("Cannot open the webcam! Exiting...")
    exit(0)

while cap.isOpened():

    flag, frame = cap.read()

    # A check to see if the frames are captured

    if not flag:
        print("Frame cannot be received! Exiting...")
        break

    # Check whether the process has been stopped/interrupted

    if cv2.waitKey(50) & 0xFF == ord("a"):
        break

    # Creating a window with a rectangular frame inside to recognise the gestures performed

    cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Recognition", (640, 480))

    cv2.rectangle(frame, (75, 75), (300, 300), (255, 255, 255), 3)

    img = frame[75:300, 75:300]

    img = img_transform(Image.fromarray(img))
    img = img.view(1, 3, 224, 224)

    pred = dl_model(img)
    class_pred = int(torch.max(pred.data, 1)[1].numpy())

    cv2.putText(frame, f"The predicted gesture is {gesture_names[class_pred]}", (200, 250), 0, 0.7,
                (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)

cap.release()
cv2.destroyAllWindows()
