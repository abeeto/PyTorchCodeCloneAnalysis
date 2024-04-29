#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:48:30 2019

@author: mzins
"""

import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from yolo import Yolo
import random
import pickle as pkl


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--dest", dest = 'dest', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()



def load_classes(file):
    with open(file, "r") as fin:
        return fin.readlines()[:-1]


# Get parameters from command line
#args = arg_parse()
class Object(object):
    pass
args = Object()
args.images = "/home/matt/Downloads/animation" #"dog-cycle-car.png"

args.batch_size = 1
args.confidence = 0.5
args.nms_thresh = 0.4
args.cfgfile = "cfg/yolov3.cfg"
args.weightsfile = "yolov3.weights"
args.reso = 416
args.dest = "det"
args.nms_thresh = 0.4

images = args.images
batch_size = args.batch_size
nms_thresh = args.nms_thresh
confidence = args.confidence


CUDA = torch.cuda.is_available()
start = 0
num_classes = 80

# Load the classes
classes = load_classes("data/coco.names")


# Set up the neural network and load weights
print("Loading network.....")
model = Yolo(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")


# get the input dimension (we assume square images)
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32


#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()

read_dir = time.time()

# get list of images to process
imlist = []
if os.path.isdir(images):
    for f in os.listdir(images):
        ext = os.path.splitext(f)[1]
        if ext == ".png" or ext == ".jpeg" or ext == ".jpg":
            imlist.append(os.path.realpath(os.path.join(images, f)))
elif os.path.exists(images):
    imlist = [os.path.realpath(images)]
else:
    print ("No file or directory with the name {}".format(images))
    exit()
print(imlist)

# create output folder
if not os.path.exists(args.dest):
    os.makedirs(args.dest)
    
    
load_batch = time.time()

# correctly resize images and create one batch per image
im_batches, orig_ims, im_dim_list = [], [], []
for f in imlist:
    b, o, d = prepare_image(f, inp_dim)
    im_batches.append(b)
    orig_ims.append(o)
    im_dim_list.append(d)
    
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
if CUDA:
    im_dim_list = im_dim_list.cuda()

leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1
    
# create the batches by concatenating images
if batch_size != 1:
    num_batches = len(imlist) // batch_size + (len(imlist) % batch_size)
    im_batches = [torch.cat(im_batches[i*batch_size:min(len(imlist), (i+1)*batch_size)]) for i in range(num_batches)]



# Process batches

write = False

start_det_loop = time.time()

output = []
for i, batch in enumerate(im_batches):
    #load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    
    
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    # filter the prediction with Non-Maxima Suppression
    prediction = filter_results(prediction, confidence, num_classes, nms=True, nms_conf=args.nms_thresh)
    

    if prediction.size(0) == 0:
        continue

    end = time.time()
    
                
    # Compute the real image indices
    prediction[:, 0] += i * batch_size
      
#    if not write:
#        output = prediction
#        write = 1
#    else:
#        print("output shape = ", output.shape)
#        print("prediction shape = ", prediction.shape)
#        output = torch.cat((output,prediction))
#        
    output.append(prediction)
    
    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output[-1] if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    
    if CUDA:
        torch.cuda.synchronize()

if len(output) == 0:
    print("No detections were made")
    exit()
    
output = torch.cat(output)
    

print("im_dim_list = ", im_dim_list)

# duplaicate im_dim_list to have one dim per detection
im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

# get the scaling factor
scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

# correct scaling of the detections
output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
output[:, 1:5] /= scaling_factor


# clamp to image size
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0]-1)
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1]-1)
    

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))
draw = time.time()



def write(x, batches, results):
    """ 
        Draw a BBox 
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

# draw all the BBoxes
list(map(lambda x: write(x, im_batches, orig_ims), output))
  

det_names = ["det/det_{}.png".format(i) for i in range(len(imlist))]
list(map(cv2.imwrite, det_names, orig_ims))

end = time.time()

print()
print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()


   