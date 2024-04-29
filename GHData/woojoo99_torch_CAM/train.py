import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.datasets as dsets

import vgg

import numpy as np
import random
import os
import shutil
import sys
import time
import collections

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

def copy_files(src, dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name))

def save_false_image(target, prediction, images):
    '''Compare target and prediction. If target != prediction, and add to dictionary as key (target, prediction).'''
    false_images_dict = collections.defaultdict(list)
    for i in range(len(target)):
        if target[i] != prediction[i]:
            temp_img = images[i]
            false_images_dict[(target[i],prediction[i])].append(temp_img)
    if len(false_images_dict) == 0:
        print("There is no false image.")
        return

    for key, val in false_images_dict.items():
        temp_img_grid = []
        for v in val:
            temp = data_transforms_inv(v)
            temp_img_grid.append(temp)
        img_grid = torchvision.utils.make_grid(temp_img_grid)
        writer.add_image('Epoch '+str(epoch)+'/'+'Label: '+CLASS_NAMES[key[0]]+', Prediction: '+CLASS_NAMES[key[1]], img_grid)



########################################################
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(777)
########################################################

### PARAMETERS
NUM_CLASSES = 4
CLASS_NAMES = ['Normal', 'Open', 'Particle', 'Remain']
N_EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
LR_DECAY = False
BATCH_NORM = True
if BATCH_NORM:
    PRETRAINED_WEIGHT_PATH = 'D:/Pretrained/torch/vgg16_bn.pth'
else:
    PRETRAINED_WEIGHT_PATH = 'D:/Pretrained/torch/vgg16.pth'
BETA1 = 0.5
#BETA1 = 0.9
BETA2 = 0.999
IMG_SIZE = 224
RANDOM_CROP = True
RANDOM_CROP_RATIO = 3/4
LOAD_SIZE = int(IMG_SIZE/RANDOM_CROP_RATIO) if RANDOM_CROP else IMG_SIZE
DATASET_PATH = 'D:/Image/dataset/train'
VALI_DATASET_PATH = 'D:/Image/dataset/validation'
EXPERIMENT_NAME = 'Torch'
DATASET_NAME = '20191028'
TRAIN_NAME = '210216'
LOG_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'log/')
CKPT_SAVE_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'ckpt/')
SOURCE_CODE_SAVE_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'src/')
TENSORBOARD_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'tensorboard/')
SAVE_FALSE_IMAGE = True
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(CKPT_SAVE_PATH, exist_ok=True)
os.makedirs(SOURCE_CODE_SAVE_PATH, exist_ok=True)

copy_files('./', SOURCE_CODE_SAVE_PATH)


vgg_cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # vgg16
    'CAM': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=NUM_CLASSES, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.myclassifier = nn.Sequential(
            nn.Dropout(), # default : p = 0.5
            nn.Linear(512, num_classes)  # output channel : 512, img size: : 28*28
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.myclassifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

### transforms (for augmentation)
class RandomRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

random_rotation = RandomRotation(angles=[90, 180, 270])

transform_mean = [0.5, 0.5, 0.5]
transform_stdv = [0.5, 0.5, 0.5]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     random_rotation,
     transforms.Resize(LOAD_SIZE),
     transforms.RandomCrop(IMG_SIZE),
     transforms.ToTensor(),
     transforms.Normalize(transform_mean, transform_stdv)
     ])

vali_transform = transforms.Compose(
    [transforms.Resize(LOAD_SIZE),
     transforms.CenterCrop(IMG_SIZE),
     transforms.ToTensor(),
     transforms.Normalize(transform_mean, transform_stdv)
     ])

data_transforms_inv = transforms.Compose(
    [transforms.Normalize(mean=list(-np.divide(transform_mean, transform_stdv)),
                          std=list(np.divide(1, transform_stdv)))
     ])


train_data = dsets.ImageFolder(root=DATASET_PATH, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

vali_data = dsets.ImageFolder(root=VALI_DATASET_PATH, transform=vali_transform)
vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=BATCH_SIZE, shuffle=False)
###classes = ('normal', 'open', 'pt', 'remain')


# define a model
model = VGG(vgg.make_layers(vgg_cfgs['CAM'], batch_norm = BATCH_NORM)).to(device)


###################################################################
###                     load pretrained weight                  ###
###################################################################
model_dict = model.state_dict()
### 0. load pretrained state dict
vgg_dict = torch.load(PRETRAINED_WEIGHT_PATH)
### 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in vgg_dict.items() if k in model_dict}
### 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
### 3. load the new state dict
model.load_state_dict(model_dict)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

if LR_DECAY:
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)  # lr decay ?

print("Start Training")
writer = tb.SummaryWriter(log_dir = TENSORBOARD_PATH, flush_secs = 30)
t_start = time.time()
total_batch_t = len(train_loader)
total_batch_v = len(vali_loader)
for epoch in range(1, N_EPOCHS+1):  # loop over the dataset multiple times
    model.train()
    running_loss_t = 0.0
    if LR_DECAY:
        lr_sche.step()

    ### training
    for X, Y in train_loader:
        ### get the inputs
        inputs = X.to(device)
        labels = Y.to(device)

        ### zero the parameter gradients
        optimizer.zero_grad()

        ### forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        ### print statistics
        running_loss_t += loss.item() / total_batch_t

    ### validation
    with torch.no_grad():
        false_images = torch.unsqueeze(torch.zeros_like(X[0]), 0)
        model.eval()
        running_loss_v = 0.0
        labels_list = []
        predictions_list = []
        img_list = []

        for X, Y in vali_loader:
            inputs = X.to(device)
            labels = Y.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            running_loss_v += loss.item() / total_batch_v

            labels_list.extend(labels.to('cpu').numpy())
            predictions_list.extend(predictions.to('cpu').numpy())

            ### for visualization of false images
            if (SAVE_FALSE_IMAGE):
                img_list.extend(X)
   
    cmat = confusion_matrix(labels_list, predictions_list)
    
    print('[Epoch:{}] Training loss = {}, Validation loss = {}'.format(epoch, running_loss_t, running_loss_v))
    print('\nConfusion Matrix')
    print(cmat)
    print('\nSummary')
    print(classification_report(labels_list, predictions_list))
    
    ### Tensorboard
    cmat_figure, ax = plot_confusion_matrix(conf_mat = cmat, colorbar = True, show_absolute = True, show_normed = True, class_names = CLASS_NAMES)
    writer.add_scalars("Accuracy", {'Accuracy':accuracy_score(labels_list, predictions_list)}, epoch)
    writer.add_scalars("Loss", {'Train Loss': running_loss_t, 'Validation Loss': running_loss_v}, epoch)
    writer.add_figure("Confusion Matrix/Validation", cmat_figure, global_step = epoch, close = False)
    if (SAVE_FALSE_IMAGE): # 적용시 epoch당 16초, 미적용시 epoch당 14초
        save_false_image(labels_list, predictions_list, img_list)

    ### Save models
    if epoch >= 200 and epoch % 10 == 0:
        torch.save(model.state_dict(), CKPT_SAVE_PATH + 'model_ep' + str(epoch) + '.pt')

writer.close()


t_end = time.time()
print('Finished Training')
print('Elapsed Time : {} seconds'.format(t_end-t_start))