# -*- coding:utf-8 -*-

import os
import random
import math
import dill 

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader

import torchtext
from torchtext.legacy import data

from classifier import Classifier
from utils import get_oversampled,  save_checkpoint, get_imbalanced
from word_correction import tokenizer

GPU_NUM = 3
#########################################################################################
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')
#########################################################################################

# ================== Parameter Definition =================
USE_CUDA = True
BATCH_SIZE = 64
SEQ_LEN = 100
IR = 10
rep = 2

emb_dim = 100
hidden_dim = 40

# ================== Dataset Definition =================
glove = torchtext.vocab.GloVe(name='6B',dim=100)

# print(len(glove.itos)) #400000
# print(glove.vectors.shape)

# TEXT = data.Field(sequential=True, batch_first=True, lower=True,init_token='<sos>', eos_token='<eos>',
#                 fix_length=SEQ_LEN+2, tokenize=tokenizer)
# LABEL = data.Field(sequential=False, batch_first=True)

# trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# TEXT.build_vocab(trainset,max_size=30000,vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
# LABEL.build_vocab(trainset)

# with open("log/TEXT.Field","wb")as f:
#      dill.dump(TEXT,f)

# with open("log/LABEL.Field","wb")as f:
#      dill.dump(LABEL,f)

# torch.save(trainset.examples,'./log/trset_orig')
# torch.save(testset.examples,'./log/tsset_orig')

trainset_examples = torch.load('./log/trset_orig')
testset_examples = torch.load('./log/tsset_orig')

with open("log/TEXT.Field","rb")as f:
     TEXT=dill.load(f)
with open("log/LABEL.Field","rb")as f:
     LABEL=dill.load(f)


trainset = data.Dataset(trainset_examples,{'text':TEXT,'label':LABEL})
testset = data.Dataset(testset_examples,{'text':TEXT,'label':LABEL})

print('훈련 샘플의 개수 : {}'.format(len(trainset)))
print('테스트 샘플의 개수 : {}'.format(len(testset)))
# print(vars(trainset[0]))
positive_subset = [i for i in trainset if vars(i)['label']=='pos']
negative_subset = [i for i in trainset if vars(i)['label']=='neg']

positive_count = int(len(negative_subset)/IR)
positive_subset = np.random.choice(positive_subset,positive_count).tolist()
trainset = positive_subset+negative_subset

positive_subset = [i for i in testset if vars(i)['label']=='pos']
negative_subset = [i for i in testset if vars(i)['label']=='neg']

positive_count = int(len(negative_subset)/IR)
positive_subset = np.random.choice(positive_subset,positive_count).tolist()
testset = positive_subset+negative_subset

trainset = data.Dataset(trainset,{'text':TEXT,'label':LABEL})
testset = data.Dataset(testset,{'text':TEXT,'label':LABEL})

VOCAB_SIZE = len(TEXT.vocab)
n_classes = 2
print('단어 집합의 크기 : {}'.format(VOCAB_SIZE))
print('클래스의 개수 : {}'.format(n_classes))
# print(TEXT.vocab.stoi)

# ================== Dataloader Definition =================

train_loader, test_loader = get_imbalanced(trainset,testset,BATCH_SIZE,TEXT)
# train_loader, test_loader = get_oversampled(trainset,testset,BATCH_SIZE,TEXT)

# batch = next(iter(train_loader))
# print([TEXT.vocab.itos[i] for i in batch[0][0]])

# ================== Model Definition =================
classifier = Classifier(num_voca=VOCAB_SIZE,emb_dim=emb_dim,hidden_dim=hidden_dim,use_cuda=USE_CUDA)
optimizer = optim.Adam(classifier.parameters(),lr=1e-2)
criterion = nn.BCEWithLogitsLoss()
if USE_CUDA:
    classifier = classifier.cuda()
    criterion = criterion.cuda()

pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
classifier.Embedding.weight.data.copy_(pretrained_embeddings)

unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
init_idx = TEXT.vocab.stoi[TEXT.init_token]
eos_idx = TEXT.vocab.stoi[TEXT.eos_token]

classifier.Embedding.weight.data[unk_idx] = torch.zeros(emb_dim)
classifier.Embedding.weight.data[pad_idx] = torch.zeros(emb_dim)
classifier.Embedding.weight.data[init_idx] = torch.zeros(emb_dim)
classifier.Embedding.weight.data[eos_idx] = torch.zeros(emb_dim)

print(classifier.Embedding.weight.data)

def accuracy(preds, y):
    preds = (torch.sigmoid(preds.data)>0.5).view(-1)
    acc = torch.sum(preds == y) / len(y)
    return acc

def compute_BA(preds, labels):

    TP = torch.logical_and(preds==1,labels==1).sum()
    FP = torch.logical_and(preds==1,labels==0).sum()
    TN = torch.logical_and(preds==0,labels==0).sum()
    FN = torch.logical_and(preds==0,labels==1).sum()

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)

    # if target.sum()!=0:

    BA = (TPR+TNR)/2
    return BA

# ================== Training Loop =================
N_EPOCH = 50
for i in range(N_EPOCH):
    classifier.train()
    train_len, train_acc, train_loss  = 0, [], []
    for batch_no, batch in enumerate(train_loader):
        optimizer.zero_grad()

        text = batch[0]
        label = batch[1]
        if USE_CUDA:
            text, label = text.cuda(), label.float().cuda()
        # label.data.sub_(1)
        
        pred = classifier(text)
        loss = criterion(pred.view(-1),label)

        predicted = torch.round(torch.sigmoid(pred.data)).view(-1)
        acc = compute_BA(predicted,label)

        train_loss.append(loss.item())
        train_acc.append(acc.item())

        loss.backward()
        optimizer.step()
    train_epoch_loss = np.mean( train_loss )
    train_epoch_acc = np.mean( train_acc )
    classifier.eval()

    with torch.no_grad():
        test_pred, test_label = [], []
        test_loss = []
        for batch in test_loader:
            text = batch[0]
            label = batch[1]
            if USE_CUDA:
                text, label = text.cuda(), label.float().cuda()
            
            pred = classifier(text)
            loss = criterion(pred.view(-1),label)
            predicted = torch.round(torch.sigmoid(pred.data)).view(-1)

            test_pred.append(predicted)
            test_label.append(label)
            test_loss.append(loss.item())
        test_pred = torch.cat(test_pred)
        test_label = torch.cat(test_label)

        acc = compute_BA(test_pred,test_label)
        print('epoch:{}/{} epoch_train_loss:{:.4f},epoch_train_acc:{:.4f}'
              ' epoch_val_loss:{:.4f},epoch_val_acc:{:.4f}'.format(i+1, N_EPOCH,
                train_epoch_loss.item(), train_epoch_acc.item(),
                np.mean(test_loss), acc))

pred_res = torch.stack((test_pred,test_label)).T
pred_res = pred_res.cpu().numpy().astype(np.int8)
np.savetxt('{:s}/rep_{:02d}_IR_{:.4f}_ROS_IMDB.csv'.format('./output',rep,IR),
pred_res,delimiter=',')
# if (i+1)%10==0:
#     save_checkpoint(acc, classifier, optimizer, i+1, 0, './log', index=True)