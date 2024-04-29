from config import config
from model.model import MobileFaceNet,Am_softmax,Arcface,Softmax
from torch.nn import DataParallel
from dataset.dataloder import Train_DATA
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from symbols.utils import Metric
import os
from  torch.autograd import Variable
import torchvision
import torch
import numpy as np

def main():
    #model pack
    model = MobileFaceNet(config.embedding_size)
    model = DataParallel(model,device_ids=config.gpu_id)
    if config.loss_type == 0:
        loss_cess = Softmax()
    elif config.loss_type == 1:
        loss_cess = Arcface(config.embedding_size,config.num_classe,config.margin_s,config.margin_m)
    else:
        loss_cess = Am_softmax(config.embedding_size,config.num_classe)
    loss_cess = DataParallel(loss_cess,device_ids=config.gpu_id)

    train_data = Train_DATA(config.train_data)
    train_loader = DataLoader(train_data,batch_size=config.batch_size,shuffle=True,num_workers=config.num_work,pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    criterion = DataParallel(criterion,device_ids=config.gpu_id)
    optimizer = optim.SGD(model.parameters(),lr=config.lr,momentum=config.momentum,weight_decay=config.weight_decay)
    optimizer = DataParallel(optimizer,device_ids=config.gpu_id)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)

    train_loss = Metric()
    train_acc = Metric()

    best_precision1 = 0
    start_epoch = 0
    fold = 0

    if config.resume:
        checkpoint = torch.load(config.model_path)
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("input data ,start run ,this is %d epoch "%(start_epoch))

    if not os.path.exists(config.model_output):
        os.makedirs(config.model_output)

    for epoch in range(start_epoch,config.end_epoch):
        scheduler.step(epoch)
        for iter,(input,target) in enumerate(train_loader):
            model.train()
            input = Variable(input)
            target = Variable(torch.from_numpy(np.array(target)).long())
            input = DataParallel(input,device_ids=config.gpu_id)
            target = DataParallel(input,device_ids=config.gpu_id)

            optimizer.zero_grad()
            embeddings = model(input)
            output = loss_cess(embeddings,target)
            loss = criterion(output,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.data.cpu().numpy()
            output = np.argmax(output,axis=1)
            label = target.data.cpu().numpy()
            acc = np.mean((label==output).astype(int))
            train_loss.updata(loss.data.cpu().numpy(),input.size(0))
            train_acc.updata(acc,input.size(0))

            if iter%20 ==0:
                print("Add valyue loss:%.3f acc:%.3f"%(train_loss.avg,train_acc.avg))

        is_best = train_acc.avg >best_precision1
        best_precision1 = max(train_acc.avg,best_precision1)
        model_savename = config.model_output+'/'+'epoch%d'%epoch+'_checkpoint.pth.tar'
        torch.save({
            "epoch":epoch+1,
            "model_name":config.model_name,
            "state_dict":model.state_dict(),
            "best_precision1":best_precision1,
            "optimizer":optimizer.state_dict(),
            "fold":fold,
            "train_loss":train_loss.avg
        },model_savename)

