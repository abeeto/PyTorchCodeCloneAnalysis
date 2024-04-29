#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# License: © 2021 Achille-Tâm GUILCHARD All Rights Reserved
# Author: Achille-Tâm GUILCHARD
# Usage: python3 inference_detection_classification.py --input <DIR> --output <DIR>

import os
import sys
import argparse
import time
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from termcolor import colored

# PyTorch Imports
from torchvision import datasets, models, transforms
import torchvision
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch

sys.path.append('/home/models/research')
sys.path.append('/home/models/research/object_detection')
sys.path.append('/home/models/research/slim')

imsize = 229

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def list_files(directory, extension):
    res_list = []
    dirFiles = os.listdir(directory)
    sorted(dirFiles)  # sort numerically in ascending order
    for f in dirFiles:
        if f.endswith('.' + extension):
            res_list.append(directory + "/" + f)
    return res_list

def get_square(image, square_size):
    height = np.size(image, 0)
    width = np.size(image, 1)
    if(height > width):
        differ = height
    else:
        differ = width
    differ += 4
    mask = np.zeros((differ, differ, 3), dtype="uint8")
    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)
    mask[y_pos:y_pos + height, x_pos:x_pos + width] = image[0:height, 0:width]
    mask = cv2.resize(mask, (square_size, square_size),
                      interpolation=cv2.INTER_CUBIC)
    return mask

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def doClassification(model_ft, crop_img):
    squared_img = get_square(crop_img, 500)
    cv2.imwrite("/tmp/squared.jpg", squared_img)
    img = image_loader("/tmp/squared.jpg")
    os.remove("/tmp/squared.jpg")

    prediction     = model_ft(img)
    ps             = torch.nn.functional.softmax(prediction)
    topk, topclass = ps.topk(1, dim=1)

    label     = labels_py[topclass.cpu().numpy()[0][0]]
    score_py  = round(topk.detach().cpu().numpy()[0][0] * 100.0, 3)
    class_img = label.lower()
    res       = {"label":label, "score": score_py}
    return res

##########################################################################
class ImageDetection():
    def __init__(self, PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as self.fid:
                self.serialized_graph = self.fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

                # self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

                # Get handles to input and output tensors
                self.ops = tf.get_default_graph().get_operations()
                self.all_tensor_names = {
                    self.output.name for self.op in self.ops for self.output in self.op.outputs}
                self.tensor_dict = {}
                for self.key in [
                    'num_detections',
                    'detection_boxes',
                    'detection_scores',
                    'detection_classes',
                        'detection_masks']:
                    self.tensor_name = self.key + ':0'
                    if self.tensor_name in self.all_tensor_names:
                        self.tensor_dict[self.key] = tf.get_default_graph(
                        ).get_tensor_by_name(self.tensor_name)
                self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                self.sess = tf.Session()

    def detect(self, img):
        # Actual detection.
        output_dict = self.sess.run(
            self.tensor_dict, feed_dict={
                self.image_tensor: np.expand_dims(
                    img, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections']     = int(output_dict['num_detections'][0])
        output_dict['detection_classes']  = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes']    = output_dict['detection_boxes'][0]
        output_dict['detection_scores']   = output_dict['detection_scores'][0]

        if 'detection_masks' in output_dict:
            output_dict['detection_masks']  = output_dict['detection_masks'][0]

        return output_dict
##########################################################################

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        #overlap = (w * h) / area[idxs[:last]]
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

flags = tf.app.flags
flags.DEFINE_string('input', '', 'Path to the input folder')
flags.DEFINE_string('output', '', 'Path to output folder')

FLAGS = flags.FLAGS

PATH_TO_FROZEN_GRAPH = "/tmp/model/frozen_inference_graph.pb"
PATH_TO_LABELS       = "/tmp/model/label.pbtxt"

PATH_TO_PT_MODEL            = "/tmp/model/resnext101_32x8d.pt"
PATH_TO_PT_MODEL_LABEL_FILE = "/tmp/model/labels.txt"

TEXT_CLASS = "item"
PATH_TO_IMAGES_DIR        = FLAGS.input
PATH_TO_IMAGES_DIR_OUTPUT = FLAGS.output

print(colored('Entries summary', 'green'))
print('  > input folder  : ' + PATH_TO_IMAGES_DIR)
print('  > output folder : ' + PATH_TO_IMAGES_DIR_OUTPUT)

PATH_TO_IMAGES = list_files(PATH_TO_IMAGES_DIR, 'jpg')

# Load AIs
img_detector = ImageDetection(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)

# Load Classification model
###########################################################################
labels_py = []

# open file and read the content in a list
with open(PATH_TO_PT_MODEL_LABEL_FILE, 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        labels_py.append(currentPlace)

# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_ft = models.resnext101_32x8d(pretrained=False)
num_ftrs = model_ft.fc.in_features
nb_classes = len(labels_py)
model_ft.fc = nn.Linear(num_ftrs, nb_classes)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load(PATH_TO_PT_MODEL, map_location='cpu'))
model_ft.eval()
###########################################################################

############
### Main ###
############

count = 1
numberofimage = len(PATH_TO_IMAGES)
PATH_TO_IMAGES.sort()

for image_path in PATH_TO_IMAGES:
    basename         = os.path.basename(image_path)
    basename_complet = basename

    print(colored("> Processing image: {} ({}/{})...".format(basename, count, numberofimage), 'red'))

    tic = time.perf_counter()
    imgcv = cv2.imread(image_path)

    if imgcv is not None:
        imgCopy         = imgcv.copy()
        img_height = np.size(imgcv, 0)
        img_width  = np.size(imgcv, 1)
        bouding_boxes   = []

        #######################################################################
        output_dict = img_detector.detect(imgcv)

        for j in range(len(output_dict['detection_boxes'])):
            output_dict['detection_boxes'][j][0] = output_dict['detection_boxes'][j][0] * img_height
            output_dict['detection_boxes'][j][1] = output_dict['detection_boxes'][j][1] * img_width
            output_dict['detection_boxes'][j][2] = output_dict['detection_boxes'][j][2] * img_height
            output_dict['detection_boxes'][j][3] = output_dict['detection_boxes'][j][3] * img_width

            bouding_box = []
            bouding_box.append(output_dict['detection_boxes'][j][0])
            bouding_box.append(output_dict['detection_boxes'][j][1])
            bouding_box.append(output_dict['detection_boxes'][j][2])
            bouding_box.append(output_dict['detection_boxes'][j][3])
            bouding_box.append(output_dict['detection_scores'][j])
            bouding_boxes.append(bouding_box)

        bouding_boxes = np.array(bouding_boxes, dtype=np.float32)

        # NMS
        bouding_boxes = non_max_suppression_fast(bouding_boxes, 0.45)
        #######################################################################

        number_of_repetition = 0
        for j in range(len(bouding_boxes)):
            ymin = bouding_boxes[j][0]
            xmin = bouding_boxes[j][1]
            ymax = bouding_boxes[j][2]
            xmax = bouding_boxes[j][3]

            h = int(ymax) - int(ymin)
            w = int(xmax) - int(xmin)

            crop_img = imgCopy[ymin:ymin + int(h), xmin:xmin + int(w)]

            res = doClassification(model_ft, crop_img)
            # toBeDisplayed = res["label"] + '(' + str(int(res["score"])) + ')'
            toBeDisplayed = res["label"]

            cv2.rectangle(imgcv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(imgcv, toBeDisplayed, (int(xmin), int(ymin) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 123, 255), 2, cv2.LINE_AA)
            
        cv2.imwrite(PATH_TO_IMAGES_DIR_OUTPUT + "/" + basename_complet, imgcv)
        count = count + 1
        toc = time.perf_counter()
        print(colored(f"...in {toc - tic:0.2f} seconds", 'green'))