# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os.path
import imagefolder
from shutil import copyfile
import convnet

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.ToPILImage(),
                                transforms.Scale(size=[32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
testset = imagefolder.ImageFolderWithPath(root='./images', transform=transform)
testloader = torch.utils.data.DataLoader(testset)

classes = ('airplane', 'automobile', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

gallery_path = './final_gallery/'
weights_file = './trained_model_weights'

# Model instance
net = convnet.ConvNet()
if os.path.isfile(weights_file):
    net.load_state_dict(torch.load(weights_file))
else:
    print('No model weight file found...')
    exit()

# Get prediction, copy file according that
for data in testloader:
    (image, label), (path, _) = data
    outputs = net(Variable(image))
    _, predicted = torch.max(outputs.data, 1)
    
    c = (predicted == label).squeeze()

    prediction = classes[predicted[0]]

    filename = os.path.basename(path[0])
    final_path = gallery_path + prediction + '/' + filename

    copyfile(path[0], final_path)

#
# ----- Whole dataset prediction stats
#
# correct = 0
# total = 0

# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# # Prediction per class
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
    
#     c = (predicted == labels).squeeze()

#     print('True label: ' + classes[labels[0]])
#     print('Predicted label: ' + classes[predicted[0]])
#     print("\r")
    
#     label = labels[0]
#     class_correct[label] += c[0]
#     class_total[label] += 1

# for i in range(10):
#     if class_total[i] > 0:
#         print('Accuracy of %5s : %2d %%' % (
#             classes[i], 100 * class_correct[i] / class_total[i]))
