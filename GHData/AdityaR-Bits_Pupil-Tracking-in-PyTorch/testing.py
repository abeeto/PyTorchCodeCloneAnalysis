# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:08:04 2020

@author: Aditya
"""

import io
import numpy as np
import cv2
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from skimage.filters import gaussian
import torch.nn as nn
import torch.nn.init as init

class Pupil_detector(nn.Module):
    def __init__(self, inplace=False):
        super(Pupil_detector, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.M1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv2_1 = nn.Conv2d(128, 128, 5)
        self.M2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv3_1 = nn.Conv2d(256, 256, 3)
        self.M3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.M4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(512, 1024, 3)
        self.M5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1024*3*3,4096)
        self.fc2 = nn.Linear(4096,5)

        #self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.M1(x)      
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2_1(x))
        x = self.M2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv3_1(x))
        x = self.M3(x)
        x = self.relu(self.conv4(x))
        x = self.M4(x)
        x = self.relu(self.conv5(x))
        x = self.M5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.abs(x) +1

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        
    #Load xml file 
from xml.dom import minidom
import os
# parse an xml file by name

def load_data_xml(data_folder, patient, split_path):
    label_file = "training.xml" if split_path == "train" else "testing.xml"
    mydoc = minidom.parse(os.path.join(data_folder,split_path,patient,label_file));

    images = mydoc.getElementsByTagName('image')
    bbox = mydoc.getElementsByTagName('box')

    list_images = []
    list_bboxes = []

    # all item attributes
    for elem in images:
        list_images.append(os.path.join(data_folder,split_path,patient,elem.attributes['file'].value))
    for elem in bbox:
       list_bboxes.append([int(elem.attributes['top'].value),int(elem.attributes['left'].value),int(elem.attributes['width'].value),int(elem.attributes['height'].value)])
    return list_images, list_bboxes
'''
        if int(elem.attributes['height'].value) <= 150: 
          top1= int(elem.attributes['top'].value)
          left1= int(elem.attributes['left'].value)
          width1= int(elem.attributes['width'].value)
          height1= int(elem.attributes['height'].value)
          #top1new= -773.068 + 16.768*top1 - 0.111*np.square(top1)
          top1new=  -53.9746 + 1.1377*top1 
          if top1new<0 :
            top1new=0
          #left1new= 1864.892 - 36.554*left1 + 0.242*np.square(left1) - 0.001*np.power(left1,3)
          #left1new= -70.407 + 1.32*left1
          left1new= -30.407 + 0.99*left1
          if left1new<0:
            left1new=0
          #width1new= 352.768 - 6.192*width1 + 0.059*np.square(width1)
          #width1new= 53.8268 + 0.86* width1
          width1new= 43.8268 + 1.20* width1
          #height1new= 352.768 - 6.192*height1 + 0.059*np.square(height1)
          height1new= 55.7215 + 1.2238* height1
          list_bboxes.append([top1new, left1new, width1new, height1new])
        else:
           list_bboxes.append([int(elem.attributes['top'].value),int(elem.attributes['left'].value),int(elem.attributes['width'].value),int(elem.attributes['height'].value)])
 '''
    import torchvision.transforms.functional as FT

def resize(image, box, dims=(224, 224), return_percent_coords=False):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)
    
    #print("box before resize", box)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.height, image.width, image.height, image.width]).unsqueeze(0)
    new_box = box / old_dims 
     # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[0], dims[1], dims[0], dims[1]]).unsqueeze(0)
        new_box = new_box * new_dims
    #print(new_box)
    return new_image, new_box.squeeze()
def transform(image, box):

    new_image, new_box = resize(image, box, dims=(224, 224))
  
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)
    

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    #new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_box

#Load data
from PIL import Image
from torch.utils.data import Dataset

class Pupil_Data(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, patients, split="train"):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split
        self.data_folder = data_folder
        self.patients = patients 

        self.images0, self.boxes0 = load_data_xml(self.data_folder,self.patients[0],self.split)
        self.images1, self.boxes1 = load_data_xml(self.data_folder,self.patients[1],self.split)
        self.images2, self.boxes2 = load_data_xml(self.data_folder,self.patients[2],self.split)
        self.images3, self.boxes3 = load_data_xml(self.data_folder,self.patients[3],self.split)
        
        self.images = self.images0 + self.images1 + self.images2 + self.images3
        self.boxes = self.boxes0 + self.boxes1 + self.boxes2 + self.boxes3
        
        assert len(self.images) == len(self.boxes)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes)
        box = self.boxes[i]
        box = torch.FloatTensor(np.array(box))  # 

        # Apply transformations
        #image = cv2.GaussianBlur(image, (5, 5), 0)
        image, box = transform(image, box)
        #print(f'shape: {image.shape}')
        

        return image, box

    def __len__(self):
        return len(self.images)
    
    %matplotlib inline
import matplotlib.pyplot as plt
data_folder = "/content/drive/My Drive/Colab Notebooks/Rootee/"
patients = ["frames_Steven","frames_Youssef","frames_Jessica2", "frames_Jessica1"]
test_dataset = Pupil_Data(data_folder, patients, split='test')
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
import torch.optim
import torch.utils.data
import matplotlib.patches as patches
#from livelossplot import PlotLosses

# Learning parameters

checkpoint = "/content/drive/My Drive/Colab Notebooks/Rootee/irisfinal_checkpoint_pupil_detector_score.pth.tar"
batch_size = 1  # batch size
# Initialize model or load checkpoint
checkpoint = torch.load(checkpoint)#, map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
#print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
optimizer = checkpoint['optimizer']

# Move to default device
model = model.to(device)
model.eval()

SMOOTH = 1e-6

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2],     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]), 1)  # xmax, ymax

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter.to(torch.float32)  / union.to(torch.float32)  # [A,B]

iou_value = 0
iouindiv=0
badcount=0
for i, (images, boxes) in enumerate(test_loader):

    # Move to default device
    #if i == 150 : break
    print(images.shape)
    images = images.to(device) 
    target_boxes = [b.to(device) for b in boxes]

    # Forward prop.
    #%timeit predicted_boxes = model(images)
    predicted_boxes = model(images)
    #print(predicted_boxes[:,:4].to(torch.int32))
    #print(target_boxes[0].unsqueeze(0).to(torch.int32))
    #print(point_form(predicted_boxes[:,:4].to(torch.int32)))
    #print(point_form(target_boxes[0].unsqueeze(0).to(torch.int32)))
    iou_value = iou_value + jaccard(point_form(predicted_boxes[:,:4].to(torch.int32)), point_form(target_boxes[0].unsqueeze(0).to(torch.int32)))
    iouindiv= jaccard(point_form(predicted_boxes[:,:4].to(torch.int32)), point_form(target_boxes[0].unsqueeze(0).to(torch.int32)))
    im = np.transpose(images.squeeze(0).cpu().numpy(),(1,2,0))
    
    # Create figure and axes
    if True: 
      # Display the image
      fig,ax = plt.subplots(1)
      ax.imshow(im)

      badcount=badcount+1
      # Create a Rectangle patch
      #print(target_boxes[0][0].to(torch.int32).item())
      
      low = predicted_boxes[0][0].to(torch.int32).item() 
      left = predicted_boxes[0][1].to(torch.int32).item()
      w = predicted_boxes[0][2].to(torch.int32).item()
      h = predicted_boxes[0][3].to(torch.int32).item()
      score = predicted_boxes[0][4].item()
      rect = patches.Rectangle((left,low),w,h,linewidth=1,edgecolor='r',facecolor='none')

      # Add the patch to the Axes
      ax.add_patch(rect)
      plt.pause(.01)
      plt.show()
      print(score)
      print(iouindiv)
    
    
    accuracy = iou_value/len(test_loader)
    accuracy = accuracy.item()*100
    
print(str(int(accuracy))+"%") 
print(f"badcount:{badcount}")   

!nvidia-smi

