#!/usr/bin/env python
# coding: utf-8


import os, torch, torchtext
    
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("PyTorch Version: ", torch.__version__)
print("torchtext Version: ", torchtext.__version__)

from models import Naive_Clf, IMDBRnn, IMDBCnn
    

def fit(epoch,model,data_loader,  phase='training'):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        
    optimizer.zero_grad()
    running_loss = 0.0
    running_correct = 0
    
    for batch_idx , batch in enumerate(data_loader):
        text, labels = batch.text , batch.label - 1
        if use_gpu:
            text, labels = text.cuda(), labels.cuda()
        with torch.set_grad_enabled(phase == 'training'):        
       
            output = model(text)
    #        print(output.shape)
            loss = F.cross_entropy(output, labels)
            
            running_loss += loss.cpu().item()
            preds = output.data.max(dim=1,keepdim=True)[1]
            running_correct += preds.eq(labels.data.view_as(preds)).cpu().sum().item()
            
            if phase == 'training':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    
    print(phase, 'loss:', loss, 'acc:', accuracy)
    return loss, accuracy

import argparse
import pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fix_length', type=int, default=150)
    parser.add_argument('--max_vocab_size', type=int, default=25000)
    parser.add_argument('--dim_embdeding', type=int, default=300, choices=[100, 300])
    parser.add_argument('--pretrained', default=True, choices=[True, False])
    parser.add_argument('--model', type=str, default='LSTM', choices=['Naive', 'CNN', 'LSTM'])
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    
    pprint(args)

    use_gpu = torch.cuda.is_available()
    
    TEXT = data.Field(lower=True, fix_length=args.fix_length, batch_first=True)
    LABEL = data.Field(sequential=False,)
    
    
    
    train, test = datasets.IMDB.splits(TEXT, LABEL)
    
        
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=args.dim_embdeding), max_size=args.max_vocab_size, min_freq=10)
    LABEL.build_vocab(train,)
    
    
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size= args.batch_size)
    train_iter.repeat = False
    test_iter.repeat = False
       
    
#    batch = next(iter(train_iter))    
    
    num_classes = 2
    
    if args.model == 'Naive':    
        model = Naive_Clf(vocab_size = TEXT.vocab.vectors.shape[0], dim_embdeding = TEXT.vocab.vectors.shape[1],
                          num_classes = num_classes, fix_length = args.fix_length)
    elif args.model == 'CNN':
        model = IMDBCnn(vocab_size=TEXT.vocab.vectors.shape[0], dim_embdeding=TEXT.vocab.vectors.shape[1], 
                          num_classes=num_classes,  fix_length=args.fix_length)
    elif args.model == 'LSTM':
        model = IMDBRnn(vocab_size = TEXT.vocab.vectors.shape[0], dim_embdeding = TEXT.vocab.vectors.shape[1],
                          num_classes=num_classes)
    
    if args.pretrained:
        model.embedding.weight.data = TEXT.vocab.vectors
        
    model.embedding.requires_grad = False
    
    if use_gpu:
        model = model.cuda()


    
    optimizer = optim.Adam(model.parameters())
        
    
    train_losses, train_accuracy = [],[]
    val_losses, val_accuracy = [],[]
    
    
    for epoch in range(args.epochs):
        print('epoch', epoch)
    
        epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')
        val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)






