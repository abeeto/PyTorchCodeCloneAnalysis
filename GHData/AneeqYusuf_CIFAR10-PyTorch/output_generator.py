# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from net import data_load, classes
import os
import numpy as np
import pandas as pd
import cv2

model = torch.load('model.sav')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

folder = '/home/aneeq/Downloads/test.rotfaces/test/'
truth_folder = '/home/aneeq/Downloads/test.rotfaces/truth/'
testset = data_load(transform=transform, folder = folder)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

test = os.listdir(folder)
output = {}

i = 0
angle_90 = 90
angle_180 = 180
angle_270 = 270
scale = 1.0
classes = list(classes)
corrected_images = []

for data in testloader:
    images = data
    outputs = model(images)
    predicted = torch.max(outputs.data, 1)[1].numpy()
    for j in predicted:
        output.update({test[i]: classes[j]})
        #read image
        f = folder+test[i]
        im = cv2.imread(f)
        #find center
        (h, w) = im.shape[:2]
        center = (w/2, h/2)
        #rotate around center
        if (j == 1):
            #left_rotated to rotate by 270, counterclockwise
            mat = cv2.getRotationMatrix2D(center, angle_270, scale)
            rotated_img = cv2.warpAffine(im, mat, (h, w))
        elif (j == 2):
            #right_rotated to rotate by 90, counterclockwise
            mat = cv2.getRotationMatrix2D(center, angle_90, scale)
            rotated_img = cv2.warpAffine(im, mat, (h, w))
        elif (j == 3):
            #upside down, to rotate by 180, counterclockwise
            mat = cv2.getRotationMatrix2D(center, angle_180, scale)
            rotated_img = cv2.warpAffine(im, mat, (h, w))
        else:
            rotated_img = im
        f2 = truth_folder + test[i][:-4]+'.png'
        cv2.imwrite(f2, rotated_img)
        corrected_images.append(rotated_img)
        i+=1
#save output as csv
df = pd.DataFrame(list(output.items()), columns=['fn', 'label'])
df.to_csv('test.preds.csv')

corrected_images = np.array(corrected_images)
f = truth_folder[:-6] + 'corrected.npy'
np.save(f, corrected_images)
print (corrected_images.shape)
