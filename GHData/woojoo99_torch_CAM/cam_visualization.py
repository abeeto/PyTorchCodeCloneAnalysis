import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import vgg

import numpy as np
import random
import os
import shutil
import sys
import time
import collections
import cv2

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

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
BATCH_NORM = True
IMG_SIZE = 224
RANDOM_CROP_RATIO = 3/4
LOAD_SIZE = int(IMG_SIZE/RANDOM_CROP_RATIO)
VISUALIZATION_DATASET_PATH = 'D:/Image/dataset/validation'
EXPERIMENT_NAME = 'Torch'
DATASET_NAME = 'dataset'
TRAIN_NAME = '210216'
CKPT_SAVE_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'ckpt/')
TENSORBOARD_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'tb_visualization/')
RESULT_PATH = os.path.join('D:/Experiments/', EXPERIMENT_NAME, DATASET_NAME, TRAIN_NAME, 'result/')

vgg_cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # vgg16
    'HDL': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
}

os.makedirs(RESULT_PATH, exist_ok=True)

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
        writer.add_image('Epoch '+str(N_EPOCHS)+'/'+'Label: '+CLASS_NAMES[key[0]]+', Prediction: '+CLASS_NAMES[key[1]], img_grid)

class VGG(nn.Module):
    def __init__(self, features, num_classes=NUM_CLASSES, init_weights=False):
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

transform_mean = [0.5, 0.5, 0.5]
transform_stdv = [0.5, 0.5, 0.5]

visualization_transform = transforms.Compose(
    [transforms.Resize(LOAD_SIZE),
     transforms.CenterCrop(IMG_SIZE),
     transforms.ToTensor(),
     transforms.Normalize(transform_mean, transform_stdv)
     ])

data_transforms_inv = transforms.Compose(
    [transforms.Normalize(mean=list(-np.divide(transform_mean, transform_stdv)),
                          std=list(np.divide(1, transform_stdv)))
     ])

visualization_data = dsets.ImageFolder(root=VISUALIZATION_DATASET_PATH, transform=visualization_transform)
visualization_loader = torch.utils.data.DataLoader(visualization_data, batch_size=BATCH_SIZE, shuffle=False)

# define & load model
model = VGG(vgg.make_layers(vgg_cfgs['HDL'], batch_norm = BATCH_NORM)).to(device)
model.load_state_dict(torch.load(CKPT_SAVE_PATH + 'model_ep' + str(N_EPOCHS) + '.pt'))

model.eval()

finalconv_name = 'features'

feature_blobs = []
def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy()) # (1, 8, 512, 14, 14)

# forward 연산 시 hook_feature 함수가 수행되도록 forward hook을 등록 
model._modules.get(finalconv_name).register_forward_hook(hook_feature)
params = list(model.parameters())
# get weight only from the last layer(linear)
weight_softmax = np.squeeze(params[-2].cpu().data.numpy()) # params[-2] : (4, 512)

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (IMG_SIZE, IMG_SIZE)
    n, c, h, w = feature_conv.shape # (8, 512, 14, 14)
    
    output_cam = []
    for i in range(n):
        cam = weight_softmax[class_idx[i]].dot(feature_conv[i].reshape((c, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
    
#for i, (image_tensor, label) in enumerate(visualization_loader):
IMG_NUM = 1
for image_tensor, label in visualization_loader:

    #image_PIL = transforms.ToPILImage()(data_transforms_inv(image_tensor[0]))
    #image_PIL.save(os.path.join(RESULT_PATH, 'img%d.png'%(i + 1)))

    image_tensor = image_tensor.to(device)

    logit = model(image_tensor) # forward
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, pred = h_x.sort(1, True)
    pred = pred[:,0] # example - label : [0, 0, 0, 0, 0, 0, 0, 0] / pred : [0, 0, 0, 0, 0, 0, 0, 0]

    CAMs = returnCAM(feature_blobs[0], weight_softmax, pred) # (8, 224, 224)
    # PIL image?

    for i in range(BATCH_SIZE):
        img = data_transforms_inv(image_tensor[i].to('cpu'))
        _, height, width = img.shape # PIL image?
        
        img = transforms.ToPILImage()(img)

        heatmap = cv2.applyColorMap(cv2.resize(CAMs[i], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + np.float32(img) * 0.5
        
        cv2.imwrite(os.path.join(RESULT_PATH, 'original_%d.png' % (IMG_NUM)), np.float32(img))
        cv2.imwrite(os.path.join(RESULT_PATH, 'cam_%d.png' % (IMG_NUM)), result)
        cv2.imwrite(os.path.join(RESULT_PATH, 'cam_only_%d.png' % (IMG_NUM)), heatmap)

        IMG_NUM += 1
    #height, width, _ = img.shape
    #heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    #result = heatmap * 0.3 + img * 0.5
    #cv2.imwrite(os.path.join(RESULT_PATH, 'cam%d.png' % (i + 1)), result)

    feature_blobs.clear()

################################################################################

#criterion = nn.CrossEntropyLoss().to(device)

#writer = tb.SummaryWriter(log_dir = TENSORBOARD_PATH, flush_secs = 30)
#total_batch_v = len(visualization_loader)

#with torch.no_grad():
#    model.eval()

#    false_images = torch.unsqueeze(torch.zeros(3, IMG_SIZE, IMG_SIZE), 0)
#    running_loss_v = 0.0
#    labels_list = []
#    predictions_list = []
#    img_list = []

#    for X, Y in visualization_loader:
#        inputs = X.to(device)
#        labels = Y.to(device)
#        outputs = model(inputs)
#        _, predictions = torch.max(outputs.data, 1)

#        loss = criterion(outputs, labels)
#        running_loss_v += loss.item() / total_batch_v

#        labels_list.extend(labels.to('cpu').numpy())
#        predictions_list.extend(predictions.to('cpu').numpy())

#        ### for visualization of false images
#        img_list.extend(X)
   
#    cmat = confusion_matrix(labels_list, predictions_list)
    
#    print('\nConfusion Matrix')
#    print(cmat)
#    print('\nSummary')
#    print(classification_report(labels_list, predictions_list))
    
#    ### Tensorboard
#    cmat_figure, ax = plot_confusion_matrix(conf_mat = cmat, colorbar = True, show_absolute = True, show_normed = True, class_names = CLASS_NAMES)
#    writer.add_scalars("Accuracy", {'Accuracy':accuracy_score(labels_list, predictions_list)}, N_EPOCHS)
#    writer.add_figure("Confusion Matrix/Validation", cmat_figure, global_step = N_EPOCHS, close = False)
#    save_false_image(labels_list, predictions_list, img_list)

#writer.close()
