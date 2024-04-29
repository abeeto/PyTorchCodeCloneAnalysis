from loss import YoloLoss
from tqdm import tqdm
import torch
import numpy as np

def run(dataloader,model,optimizer,device,cell_size=7,num_boxes=2,num_classes=3):
    losses = []
    ious = []
    loader = tqdm(dataloader)
    for data,target in loader:
        optimizer.zero_grad()
        data,target = data.to(device),target.to(device)
        pred = model(data)
        loss,class_loss,conf_loss,box_loss,iou = YoloLoss(pred,target,cell_size=cell_size,num_boxes=num_boxes,num_classes=num_classes,device=device)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        ious.append(iou.item())

        loader.set_postfix(train_loss_batch=loss.item(),class_loss=class_loss.item(),conf_loss=conf_loss.item(),box_loss=box_loss.item(),train_iou_batch=iou.item())
    
    return sum(losses)/len(losses),sum(ious)/len(ious)