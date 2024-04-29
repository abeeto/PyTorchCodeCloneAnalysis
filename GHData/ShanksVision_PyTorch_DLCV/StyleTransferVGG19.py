# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:34:49 2020

@author: shankarj


need to fix the retrain_graph=True issue in total_loss.backward() call
"""
import torch as pt
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
import torch.nn.functional as F
import sys
import pytorch_model_summary as summary
import PIL

#import images 

content_image = PIL.Image.open('../Data/style_transfer/Roses.jpg')
style_image = PIL.Image.open('../Data/style_transfer/StarryNight.jpg')

model = tv.models.vgg19(True)
vgg19_features = model.features
gpu = pt.device("cuda:0")

for param in vgg19_features.parameters():
    param.requires_Grad = False
    
vgg19_features.to(gpu)

pre_proc = tv.transforms.Compose([tv.transforms.Resize(400),
                                  tv.transforms.ToTensor(),
                                  tv.transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))])

content_tensor = pre_proc(content_image).unsqueeze(0).to(gpu)
style_tensor = pre_proc(style_image).unsqueeze(0).to(gpu)


def get_features(image, model):
    layers = {'0': 'layer0',
              '5': 'layer5',
              '10': 'layer10',
              '19': 'layer19',
              '21': 'layer21',
              '28': 'layer28'}
    
    features = {}
    
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    
    return features

content_features = get_features(content_tensor, vgg19_features)
style_features = get_features(style_tensor, vgg19_features)   

def gram_matrix(tensor):
    _, d, h, w = tensor.shape
    tensor = tensor.view(d, h*w)
    gram = pt.mm(tensor, tensor.t())
    return gram


style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}            

style_weights = {'layer0': 1,
                 'layer5': 0.75,
                 'layer10': 0.25,
                 'layer19': 0.25,                 
                 'layer28': 0.15}

content_weight = 1  #alpha
style_weight = 1e4  #beta

target_tensor = content_tensor.clone().requires_grad_(True).to(gpu)

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)

    return image

plot_every = 400
optimizer = pt.optim.Adam([target_tensor], lr=0.003)
epochs = 12000

height, width, channels = im_convert(target_tensor).shape
image_array = np.empty(shape=(plot_every, height, width, channels))
capture_frame = epochs/plot_every
counter = 0

for i in range(1, epochs+1):
    target_features = get_features(target_tensor, vgg19_features)
    content_loss = pt.mean((target_features['layer21'] - content_features['layer21'])**2)
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * pt.mean((target_gram - style_gram)**2)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    #total_loss.backward()
    if counter == 0:
        total_loss.backward(retain_graph=True)
    else:
        total_loss.backward()
    optimizer.step()
    
    if  i % plot_every == 0:
        print('Total loss: ', total_loss.item())
        print('Iteration: ', i)
        plt.imshow(im_convert(target_tensor))
        plt.axis("off")
        plt.show()
    
    if i % capture_frame == 0:
        image_array[counter] = im_convert(target_tensor)
        counter = counter + 1