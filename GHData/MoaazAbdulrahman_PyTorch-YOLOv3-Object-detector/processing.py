import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from darknet import Darknet
import time
from moviepy.editor import VideoFileClip

cfg_file = './cfg/yolov3.cfg'
weight_file = './weights/yolov3.weights'
namesfile = 'data/coco.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)

# Set the IOU threshold. Default value is 0.4
iou_thresh = 0.4
# Set the NMS threshold. Default value is 0.6
nms_thresh = 0.6

plt.rcParams['figure.figsize'] = [24.0, 14.0]

def processing(frame):
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (m.width, m.height))
    # Detect objects in the image
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    # Print the objects found and the confidence level
    #print_objects(boxes, class_names)

    # Plot the image with bounding boxes and corresponding object class labels
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    image = plot_boxes(original_image, boxes, class_names, plot_labels=False, count=True)

    return image

video_output = './output/people_out.mp4'
print('reading...')
clip = VideoFileClip('./images/people.mp4')
print('processing...')
test_clip = clip.fl_image(processing)
print('writing...')
test_clip.write_videofile(video_output, audio=False)

'''
img = cv2.imread('./images/desk.jpeg')
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(original_image, (m.width, m.height))
'''




