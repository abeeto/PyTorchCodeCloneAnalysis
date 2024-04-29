import torch 
import torch.nn as nn
import numpy as np
import cv2 
from utils.utils import *
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from utils.YOLODataLoader import *
from model.YOLO import YOLO


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='YOLO V3 Dection')
    parser.add_argument('--img_size',type=int,default=416,help='size of input image')   
    parser.add_argument('--n_cpu',type=int,default=4,help='number of cpu threads creating dataloader')
    parser.add_argument('--img_path',type=str,default='imgs/detect.txt',help='training data path')
    parser.add_argument('--save_path',type=str,default='output/',help='detection data path')
    parser.add_argument('--class_path', type=str, default='data/handsup.names', help='path to class label file')
    parser.add_argument('--data_config_path',type=str,default='cfg/handsup.data',help='location of data config file')
    parser.add_argument('--model_config_path',type=str,default='cfg/handsup.cfg',help='location of model config file')
    parser.add_argument('--weights_path',type=str,default='weights/handsup.weights',help='locaiton of weights file')
    parser.add_argument('--confidence',type=float,default=0.5,help='object confidence threshold')
    parser.add_argument('--nms_thresh',type=float,default=0.4,help='IOU threshold for non-maxumum suppression')
    parser.add_argument('--use_GPU',type=bool,default=True,help='if use GPU for training')
    parameters = parser.parse_args()

    CUDA = torch.cuda.is_available() and parameters.use_GPU

    data_config = parseDataConfig(parameters.data_config_path)
    num_classes = int(data_config['classes'])

    classes = loadClass(parameters.class_path) 

    print("Loading network.....")
    model = YOLO(parameters.model_config_path,num_classes)
    model.loadModel(parameters.weights_path)
    
    model.net_info["height"] = parameters.img_size

    if CUDA:
        model.cuda()
    
    model.eval()
    
    dataloader = getDataLoader(parameters,parameters.img_path,is_train=False)

    imgs = []
    img_detections = []

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        if CUDA:
            batch = input_imgs.cuda()
        
        with torch.no_grad():
            prediction = model(batch, CUDA)
            prediction = nonMaxSuppression(prediction, parameters.confidence, num_classes, True, parameters.nms_thresh)
            imgs.append(img_paths)
            img_detections.append(prediction)

    cmap = plt.get_cmap('tab20b')

    for img_i, (path, outputs) in enumerate(zip(imgs, img_detections)):
        count = Counter()
        output_text = ""

        img = np.array(Image.open(path[0]))
        dim = max(img.shape)
        scaling_factor = min(parameters.img_size/dim,1)

        if type(outputs) == int:
            continue

        for output in outputs:
            count[classes[int(output[-1])]] += 1
            output[1] -= (parameters.img_size - scaling_factor*img.shape[1])/2
            output[3] -= (parameters.img_size - scaling_factor*img.shape[1])/2
            output[2] -= (parameters.img_size - scaling_factor*img.shape[0])/2
            output[4] -= (parameters.img_size - scaling_factor*img.shape[0])/2
            output[1:5] /= scaling_factor

            c1 = tuple(output[1:3].int())
            c2 = tuple(output[3:5].int())
            cls = int(output[-1])
            label = "{0}".format(classes[cls])
            cv2.rectangle(img, c1, c2,[0,0,255], 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2,[0,0,255], -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)


        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,30)
        fontScale              = 0.5
        fontColor              = (0,0,255)
        lineType               = 1

        for key, value in count.items():
            string = '{}: {}'.format(key, value)
            output_text += string
            output_text += '\n'

        for i, line in enumerate(output_text.split('\n')):
                y = 15 + i*15
                cv2.putText(img,line, (10,y), font, fontScale,fontColor,lineType)

        output = cv2.merge([img[...,2],img[...,1],img[...,0]])
        cv2.imwrite(parameters.save_path+str(img_i)+'.jpg', output)
    
