
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *
from darknet import Darknet

# Set the location and name of the cfg file
cfg_file = './cfg/yolov3.cfg'

# Set the location and name of the pre-trained weights file
weight_file = './weights/yolov3.weights'

# Set the location and name of the COCO object classes file
namesfile = 'data/coco.names'

# Load the network architecture
model = Darknet(cfg_file)

# Load the pre-trained weights
model.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)

# Load the image
img = cv2.imread('./test_images/surfer.jpg')

# Convert the image to RGB
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# We resize the image to the input width and height of the first layer of the network.
resized_image = cv2.resize(original_image, (model.width, model.height))

# Display the images
plt.imshow(original_image)
plt.axis("off")
plt.show()

# Set the NMS threshold
nms_thresh = 0.6

# Set the IOU threshold
iou_thresh = 0.4

# Detect objects in the image
boxes = detect_objects(model, resized_image, iou_thresh, nms_thresh)

# Print the objects found and the confidence level
print_objects(boxes, class_names)

# Plot the image with bounding boxes and corresponding object class labels
plot_boxes(original_image, boxes, class_names, plot_labels=True)
