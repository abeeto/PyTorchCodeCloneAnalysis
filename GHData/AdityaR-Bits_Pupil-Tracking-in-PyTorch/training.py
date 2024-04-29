# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:11:18 2020

@author: Aditya
"""

import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Super Resolution model definition in PyTorch
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
        '''if int(elem.attributes['height'].value) <= 150: 
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
          list_bboxes.append([round(top1new), round(left1new), round(width1new), round(height1new)])
        else:
           list_bboxes.append([int(elem.attributes['top'].value),int(elem.attributes['left'].value),int(elem.attributes['width'].value),int(elem.attributes['height'].value)])
        '''
    print(len(list_images))
    print(len(list_bboxes))
    return list_images, list_bboxes

import torchvision.transforms.functional as FT
import random

def random_crop(image, box):
  image_h = image.size(1) # 350
  image_w = image.size(2) # 400
  #print("image.size",image.size())
  #print("box.size",box)
  #crop coordinates

  top = random.randint(0, box[0].item())
  left = random.randint(0, box[1].item())
  try:
    bottom = random.randint(box[0].item()+box[3].item(), image_h)
  except:
    bottom = image_h   
  try:
    right = random.randint(box[1].item()+box[2].item(), image_w)
  except:
    right = image_w
  #print(top, left, bottom, right)
  #crop image
  #print("top,bottom",(top,bottom))
  #print("left,right",(left,right))
  new_image = image[:, top:bottom, left:right]
  #print("done")
  #new box coords
  new_box = box - torch.FloatTensor([top, left, 0, 0])
  #old_dims = torch.FloatTensor([image_h, image_w, 1, 1])
  #ratio = box / old_dims 
    # percent coordinates
  

  #new_dims = torch.FloatTensor([new_image.size(1), new_image.size(2), 1, 1])
  #new_box = ratio * new_dims
  #print(new_box)
  return new_image, new_box


def expand(image, boxes):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    Helps to learn to detect smaller objects.
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 2
    scale = random.uniform( 1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor([150/255,30/255,135/255])  # (3) [0.485, 0.456, 0.406]
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([top, left, 0, 0])  # (n_objects, 4), n_objects is the no. of objects in this image 
    return new_image, new_boxes

def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image

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
    old_dims = torch.FloatTensor([image.height, image.width, image.width, image.height])
    new_box = box / old_dims 
     # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[0], dims[1], dims[0], dims[1]])
        new_box = new_box * new_dims
        #if (new_box[0][2]>100 and new_box[0][3]>112):
          #new_box = new_box + torch.FloatTensor([0, 0, -8, 10])
        #elif(new_box[0][2]>70 and new_box[0][3]>70):
          #new_box = new_box + torch.FloatTensor([0, 0, -4, 7])
          #new_box = new_box + torch.FloatTensor([0, 0, -2, 2]) 
    return new_image, new_box

def transform(image, new_box):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_image = photometric_distort(image)
    new_image = FT.to_tensor(new_image)
    if random.random() < 0.5:
      new_image, new_box = random_crop(new_image, new_box)
    if random.random() < 0.5:
      new_image, new_box = expand(new_image, new_box)
    new_image = FT.to_pil_image(new_image)
    new_image, new_box = resize(new_image, new_box, dims=(224, 224))
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    #new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_box.squeeze()


!nvidia-smi

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
        #self.images4, self.boxes4 = load_data_xml(self.data_folder,self.patients[4],self.split)
        #self.images5, self.boxes5 = load_data_xml(self.data_folder,self.patients[5],self.split)
        '''self.images6, self.boxes6 = load_data_xml(self.data_folder,self.patients[6],self.split)
        self.images7, self.boxes7 = load_data_xml(self.data_folder,self.patients[7],self.split)
        self.images8, self.boxes8 = load_data_xml(self.data_folder,self.patients[8],self.split)'''
        self.images = self.images0 + self.images1 + self.images2 + self.images3# + self.images4 #+self.images5#+self.images6+self.images7+self.images8
        self.boxes = self.boxes0 + self.boxes1 + self.boxes2 + self.boxes3 #+ self.boxes4 #+ self.boxes5 #+ self.boxes6 + self.boxes7 + self.boxes8
        assert len(self.images) == len(self.boxes)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes)
        box = self.boxes[i]
        box = torch.FloatTensor(np.array(box))   

        # Apply transformations
        image, box = transform(image, box)

        return image, box

    def __len__(self):
        return len(self.images)
    
    from google.colab import drive
drive.mount('/content/drive')

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
data_folder = "/content/drive/My Drive/Colab Notebooks/Rootee/"
patients = [ "pupil_frames_brian_relabeled", "pupil_frames_Jessica3_relabeled", "pupil_frames_Merry_relabeled/frames_Merry_relabeled", "pupil_frames_Soul_relabeled"]#"frames_1","frames_2","frames_brian","frames_Gray","frames_Jessica3","frames_Soul","frames_Merry","frames_Youssef2", "pupil_dataset/frames_Brian"]
train_dataset = Pupil_Data(data_folder, patients, split='train')
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=1,
                                           pin_memory=True) 
#Training Loss
#torch.set_grad_enabled(True)
mse_loss = nn.MSELoss().to(device)
l1_loss = nn.L1Loss().to(device)
class Pupil_Loss(nn.Module):
    def forward(self,predicted_boxes, target_boxes, image):
        #torch.set_grad_enabled(True)
        predicted_pupil = image[:,:,int(predicted_boxes[0][0].item()):int(predicted_boxes[0][0].item())+int(predicted_boxes[0][3].item()),int(predicted_boxes[0][1].item()):int(predicted_boxes[0][1].item())+int(predicted_boxes[0][2].item())]
        target_pupil = image[:,:,int(target_boxes[0][0].item()):int(target_boxes[0][0].item())+int(target_boxes[0][3].item()),int(target_boxes[0][1].item()):int(target_boxes[0][1].item())+int(target_boxes[0][2].item())]
        
        #if (predicted_pupil.size(2)==0 or predicted_pupil.size(3)==0):
            #predicted_pupil = torch.zeros(1,3,10,10)
        predicted_pupil = torch.nn.functional.interpolate(predicted_pupil,(int(target_pupil.size(2)),int(target_pupil.size(3))))
        #plt.imshow(np.transpose(predicted_pupil.squeeze().numpy(), (1, 2, 0)))
        #plt.show()
        #plt.imshow(np.transpose(target_pupil.squeeze().numpy(), (1, 2, 0)))
        #plt.show()
        #print(predicted_pupil.size(),target_pupil.size())
        bbox_loss = mse_loss(predicted_boxes[0][:4],target_boxes[0].unsqueeze(0).to(device))
        roi_loss = mse_loss(predicted_pupil,target_pupil)
        roi_loss = torch.tensor(roi_loss,requires_grad=True)
        score_loss = mse_loss(roi_loss*100,predicted_boxes[0][4])
        #print(predicted_boxes[0][4])
        return bbox_loss + score_loss #+ roi_loss*100 

import time
import torch.optim
import torch.utils.data
#from livelossplot import PlotLosses


def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = '/content/drive/My Drive/Colab Notebooks/Rootee/pupilfinal_checkpoint_pupil_detector_score.pth.tar'
    torch.save(state, filename)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#liveloss = PlotLosses()
def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    logs = {}
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device) 
        target_boxes = [b.to(device) for b in boxes]

        # Forward prop.
        predicted_boxes = model(images) 

        # Loss
        loss = criterion(predicted_boxes, target_boxes, images)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)
        #logs['log loss'] = loss.item()

        start = time.time()

        # Print status
        if i % print_freq == 0:
            #liveloss.update(logs)
            #liveloss.draw()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

# Data parameters
#data_folder = "/content/drive/My Drive/Colab Notebooks/Rootee/"
#patients = ["frames_brian","frames_Gray","frames_Soul","frames_Merry","pupil_dataset/frames_Brian"]


# Learning parameters
checkpoint = "/content/drive/My Drive/Colab Notebooks/Rootee/pupilfinal_checkpoint_pupil_detector_score.pth.tar"  # path to model checkpoint, None if none
#checkpoint = None
batch_size = 1  # batch size
epochs = 1700 # number of iterations to train
workers = 1  # number of workers for loading data in the DataLoader
print_freq = 5  # print training status every __ batches
lr = 1e-6  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

# Initialize model or load checkpoint
if checkpoint is None:
    start_epoch = 0
    model = Pupil_detector()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr, weight_decay=weight_decay)

else:
    checkpoint = torch.load(checkpoint)#, map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1
    #print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']

# Move to default device
model = model.to(device)
criterion = Pupil_Loss().to(device)

# Custom dataloaders
train_dataset = Pupil_Data(data_folder, patients, split='train')
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True) 

# Epochs
for epoch in range(start_epoch, epochs):

    # One epoch's training
    #if (train(train_loader=train_loader,model=model,criterion=criterion, optimizer=optimizer, epoch=epoch) == False) : break
    train(train_loader=train_loader,model=model,criterion=criterion, optimizer=optimizer, epoch=epoch)
    # Save checkpoint
    if (epoch % 5 == 0):
        save_checkpoint(epoch, model, optimizer)
