# -*- coding: utf-8 -*-
"""
=================================================================================================================================
Title: capture_images.py
Description: Capturing images of the predefined gestures using the webcam. A window will be opened to capture <num_images> images
of the <gesture_label> gesture. Press "s" to start the recording, press "d" to pause the recording and press "a" to stop
the recording and close the window
Syntax: python capture_images.py <num_images> <gesture_label> <set>
==================================================================================================================================
"""

__author__ = "Aparajit Balaji"

# Importing necessary modules

import os
import sys
import cv2

import warnings
warnings.filterwarnings("ignore")

# The parameters gesture_label, num_images, and the set are accepted from the code run in the terminal to execute the file

try:
    num_images = sys.argv[1]
    gesture_label = sys.argv[2]
    data_set = sys.argv[3]
except:
    print("Arguments missing! Please execute the file based on the below mentioned syntax...")
    print("\n python capture_images.py <num_images> <gesture_label> <data_set>")
    exit(0)

# Setting the path parameters, and creating file directories to store the images of each gesture

img_directory = "images"
set_directory = os.path.join(img_directory, data_set)
gesture_directory = os.path.join(set_directory, gesture_label)

try:
    os.mkdir(img_directory)
except FileExistsError:
    print("Root directory already exists at {}".format(img_directory))
    pass

try:
    os.mkdir(set_directory)
except FileExistsError:
    print(f"{data_set} set already exists!")
    pass

try:
    os.mkdir(gesture_directory)
except FileExistsError:
    print("The gestures directory exists in {}".format(gesture_directory))
    print("\n The images captured will be stored in this directory")

# Accessing the webcam of the system

cap = cv2.VideoCapture(0)

start = False
cnt = 0

# Code to capture the frames

# Check if webcam is available

if not cap.isOpened():
    print("Cannot open the webcam! Exiting...")
    exit(0)

while cap.isOpened():
    flag, frame = cap.read()

    # A check to see if the frames are captured

    if not flag:
        print("Frame cannot be received! Exiting...")
        break

    # Check for the number of images captured

    if cnt == int(num_images):
        break

    # Starting the capturing process

    if cv2.waitKey(50) == ord("s"):
        start = True
    
    # Exting the capturing process

    if cv2.waitKey(50) & 0xFF == ord("a"):
        break
    
    # Pausing the capturing process
    
    if cv2.waitKey(50) == ord("d"):
        cv2.waitKey(-1)

    # Creating a webcam window

    cv2.namedWindow("Image Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Collector", (640, 480))
    cv2.putText(frame, "{} images collected...".format(cnt), (400, 400), 0, 0.7,
                (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Image Collector", frame)

    # Creating a rectangular frame
    
    cv2.rectangle(frame, (75, 75), (300, 300), (255, 255, 255), 3)

    # Capturing and storing the images
    
    if start:
        img = frame[75:300, 75:300]
        img_path = os.path.join(gesture_directory, "{}.jpg".format(cnt + 1))
        cv2.imwrite(img_path, img)
        cnt += 1

# Displaying the number of images captured in the console

print("\n {} images have been captured and saved in the directory {}".format(cnt, gesture_directory))

# Closing the webcam window after the capturing is complete

cap.release()
cv2.destroyAllWindows()