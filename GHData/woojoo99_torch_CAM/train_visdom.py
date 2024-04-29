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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import visdom
### python -m visdom.server
vis = visdom.Visdom()
vis.close(env="main")
vis.close(env="false")

def copy_files(src, dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name))

########################################################
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda:0':
    torch.cuda.manual_seed_all(777)
########################################################

### PARAMETERS
NUM_CLASSES = 4
N_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.00001
LR_DECAY = False
BATCH_NORM = False
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
TRAIN_NAME = '210205'
LOG_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'log/')
CKPT_SAVE_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'ckpt/')
SOURCE_CODE_SAVE_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'src/')
VISUALIZATION = True

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

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     random_rotation,
     transforms.Resize(LOAD_SIZE),
     transforms.RandomCrop(IMG_SIZE),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

vali_transform = transforms.Compose(
    [transforms.Resize(LOAD_SIZE),
     transforms.CenterCrop(IMG_SIZE),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

loss_plt = vis.line(Y=torch.Tensor([1,1]).zero_(), opts=dict(title='Loss Tracker', legend = ['Training loss', 'Validation loss'], showlegend=True), env="main")
print("Start Training")
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
        model.eval()
        running_loss_v = 0.0
        for X, Y in vali_loader:
            inputs = X.to(device)
            labels = Y.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            running_loss_v += loss.item() / total_batch_v

            ### for visualization of false images
            if (VISUALIZATION and epoch > 40):
                false = (labels != predictions)
                false_list = []
                outputs_list = []
                predictions_list = []
                for i in range(0, BATCH_SIZE):
                    if false[i]:
                        # PIL image [0.0 ~ 1.0]
                        vis.image(X[i]*0.5 + 0.5, env="false",
                                  opts=dict(title = "Epoch [{}]:".format(epoch),
                                      caption="label : class {}, prediction : class {}".format(Y[i], predictions[i])))            
    
    print('[Epoch:{}] Training loss = {}, Validation loss = {}'.format(epoch, running_loss_t, running_loss_v))
    # loss tracker
    vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([running_loss_t]), win=loss_plt, name='Training loss', update='append', env="main")
    vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([running_loss_v]), win=loss_plt, name='Validation loss', update='append', env="main")




t_end = time.time()
print('Finished Training')
print('Elapsed Time : {} seconds'.format(t_end-t_start))

labels_list = []
predictions_list = []
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for data in vali_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs.data, 1)

        labels_list.extend(labels.to('cpu').numpy())
        predictions_list.extend(predictions.to('cpu').numpy())

        total += labels.size(0)
        
        correct += (predictions == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

print(confusion_matrix(labels_list, predictions_list))
print(classification_report(labels_list, predictions_list))

f = open(LOG_PATH + 'log.txt', 'w')
sys.stdout = f
print(confusion_matrix(labels_list, predictions_list))
print(classification_report(labels_list, predictions_list))
sys.stdout = sys.__stdout__
f.close()