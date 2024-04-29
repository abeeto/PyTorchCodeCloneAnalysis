import numpy as np
import pickle, os
import skimage.io
import matplotlib.pyplot as plt

from model import LanguageModel, VQA_FeatureModel, VGG16
from data_loader import ImageFeatureDataset
from data_utils import change, preprocess_text

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torchvision.models as models


with open('dumps/val_features_vgg16.pkl', 'rb') as f:
    v = pickle.load(f)
paths = [x[1] for x in v]


vgg16 = VGG16()
print('Loaded VGG16 Model')
vgg16 = vgg16.eval()

img_transform = transforms.Compose([
                                     transforms.Resize(( 224, 224 )),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                     ])

def get_feature(path):

    img = skimage.io.imread(path)
    
    if(len(img.shape) == 2):
        img = np.array([img]*3)
        img = np.moveaxis(img, 0, 2)

    img = F.to_pil_image(img)
    img = img_transform(img)
    img = img.unsqueeze(0)
    output = vgg16(img)
    output = output.view(-1)

    return output.detach()






with open('dumps/input_embedding.pkl', 'rb') as f:
    input_embedding = pickle.load(f)
print('input_embedding loaded')


with open('dumps/idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)
print('idx2word loaded')


model = torch.load('checkpoint/model_vgg16.pth')
model = model.eval()
print('model loaded')

def predict_image(path, question, model=model, idx2word=idx2word, input_embedding=input_embedding):

    
    input_seq_len = 21
    embedding_size = 50

    X1 = get_feature(path)
    
    question = preprocess_text(question)
    padding = ['<pad>']*(input_seq_len-len(question))
    question = question + padding

    X2 = np.zeros((input_seq_len, embedding_size))
    for i in range(input_seq_len):
        if question[i] not in input_embedding.keys():
            question[i] = '<unk>'
        X2[i] = input_embedding[question[i]]

    X2 = torch.from_numpy(X2).float() 

    X1 = X1.unsqueeze(0)
    X2 = X2.unsqueeze(0)
    
    if torch.cuda.is_available():
        X1 = X1.cuda()
        X2 = X2.cuda()
        model = model.cuda()

    output = model(X1, X2)
    index = output.max(-1)[1].item()

    return idx2word[index]



def display(path):

    for i in range(len(v)):
        _, p, q, a = v[i]
        if p == path:
            print('question : ', q)
            print('answer : ', a)
            break

    img = skimage.io.imread(path)
    plt.imshow(img)
    plt.show(block=False)
