import os
import json
configs = json.load(open("config.json"))

os.chdir(configs['working_dir'])

from engine import train_one_epoch, evaluate
import utils
from load_image import Dataset
import torch
import numpy as np

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch import nn

# import transforms as T
from torchvision import datasets, models, transforms

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()
mytransform = transforms.Compose([
        transforms.ToTensor()
    ])


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
                                                

device = torch.device(configs["device"])

# our dataset has two classes only - background and person
# use our dataset and defined transformations
dataset = Dataset(configs["train_data_path"],mytransform)
dataset_test = Dataset(configs["test_data_path"],mytransform)

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:])
dataset_test = torch.utils.data.Subset(dataset_test, indices[:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=6,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=6,
    collate_fn=utils.collate_fn)


#move model to the right device
if configs["deactivate_batchnorm"] == "True":
    model.apply(deactivate_batchnorm)

if configs["pretrained_model"] != "":
    checkpoint = torch.load(configs["pretrained_model"])
    model.load_state_dict(checkpoint,strict=True)


model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=configs["learning_rate"],
                            momentum=0.9, weight_decay=0.0001)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=configs["decay_step"],
                                                gamma=configs["decay_rate"])

# let's train it for 10 epochs
#num_epochs = 20
epoch_start = 0
epoch_end = configs["epoch"]
best_loss=1000
train_step=0
if configs["continue_training"] == "True":
    try:
        log = np.loadtxt('log.txt',delimiter= ',')
        epoch_start = int(log[-1][0])
        train_step = int(log[-1][1])
        checkpoint = torch.load(open("check_points_log.txt", "r").readlines()[-1]).splitlines()
        model.load_state_dict(checkpoint,strict=True)

    except:
        pass


#for epoch in range(num_epochs):
for epoch in range(epoch_start, epoch_end):
    #pass
    # train for one epoch, printing every 10 iterations
    train_step=train_one_epoch(model, optimizer, data_loader, device, epoch, best_loss,train_step,print_freq=10)

    # # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model,data_loader_test,torch.device('cpu'))

