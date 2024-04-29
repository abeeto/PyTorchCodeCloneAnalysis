import os
import json
configs = json.load(open("config.json"))

os.chdir(configs['working_dir'])
import torch
import cv2
import numpy as np
from PIL import Image , ImageDraw
import matplotlib.pyplot as plt
import utils
#from frcnn import frcnn_model
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from skimage.measure import find_contours
import random
import colorsys
from torchvision import datasets, models, transforms
from torch import nn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--model_path', help='test config file path')
    parser.add_argument('--cuda_device', type=str,help='cuda device')
    parser.add_argument(
        '--video_path', type=str, help='camera device id')
    parser.add_argument(
        '--score_thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def apply_mask(image, mask, color, alpha=0.5):
  """Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] *
                              (1 - alpha) + alpha * color[c],
                              image[:, :, c]
                              )
  return image

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

def find_rec(points):
    sort_x = sorted(points, key=lambda x: x[0])
    x1=sort_x[0][0]
    x2=sort_x[-1][0]
    sort_y = sorted(points, key=lambda x: x[1])
    y1=sort_y[0][1]
    y2=sort_y[-1][1]
    tmp=0
    if x1 > x2: 
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2: 
        tmp = y2
        y2 = y1
        y1 = tmp
    return [int(x1),int(y1),int(x2),int(y2)]
@torch.no_grad()
def main(args):
        
    #img_path='./icremation_imgs/demo_imgs'
    device = torch.device(args.cuda_device)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    PATH=args.model_path
    transform1 = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ])
    num_classes = int(configs["number_of_class"])+1  # class + background
    # # get number of input features for the classifier
    
    if configs["model_name"] == "mask_rcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif configs["model_name"] == "faster_rcnn": 
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # # replace the classifier with a new one, that has
    # # num_classes which is user-defined
    num_classes = int(configs["number_of_class"])+1  #  class  + background
    # # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if configs["model_name"] == "mask_rcnn":
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                            hidden_layer,
                                                            num_classes)
                                                
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)
    model.apply(deactivate_batchnorm)
    model.to(device)
    model.eval()
    cap = cv2.VideoCapture(args.video_path)
    while True:
        success,image = cap.read()
        if not success:break
        image1 = Image.fromarray(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB))
        image_tensor=transform1(image1)  
        
        image_tensor=image_tensor.unsqueeze(0).to(device)

        predictions = model(image_tensor)
        masks=predictions[0]["masks"]
        scores=predictions[0]["scores"]
        class_id = predictions[0]["labels"]
        boxes = predictions[0]["boxes"]
        pts=[]
        for i,mask in enumerate(masks):
            score = scores.data[i]
            if score < args.score_thr:continue
            bb_box = boxes.data[i]
            if configs["model_name"] == "mask_rcnn":

                mask = mask.cpu().detach().numpy()
                mask[mask>=0.6]=1

                mask = np.reshape(mask,(mask.shape[1],mask.shape[2],mask.shape[0])).astype(np.uint8)

                color =  (0,255,0)

                contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

                cv2.drawContours(image,contours,0,color,4) 
                mask_copy = np.reshape(mask,(mask.shape[0],mask.shape[1])).astype(np.uint8)
                image=apply_mask(image,mask_copy,color)

            cv2.rectangle(image,(bb_box[2],bb_box[1]),(bb_box[0],bb_box[3]),color,2)


            #cv2.imwrite("result.png",image)
        show_image = cv2.resize(image.copy(),(720,480))
        cv2.imshow('image',show_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

if __name__ == '__main__':
     args = parse_args()
     main(args)



    
