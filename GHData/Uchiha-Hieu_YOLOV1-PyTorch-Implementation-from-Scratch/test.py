from loss import YoloLoss
from tqdm import tqdm
import torch
import numpy as np

def run(dataloader,model,device,cell_size=7,num_boxes=2,num_classes=3):
    model.eval()
    losses = []
    ious = []
    loader = tqdm(dataloader)
    with torch.no_grad():
        for data,target in loader:
            data,target = data.to(device),target.to(device)
            pred = model(data)
            loss,class_loss,conf_loss,box_loss,iou = YoloLoss(pred,target,cell_size=cell_size,num_boxes=num_boxes,num_classes=num_classes,device=device)        
            losses.append(loss.item())
            ious.append(iou.item())
            loader.set_postfix(test_loss_batch=loss.item(),class_loss=class_loss.item(),conf_loss=conf_loss.item(),box_loss=box_loss.item(),test_iou_batch=iou.item())
    
    model.train()
    return sum(losses)/len(losses),sum(ious)/len(ious)