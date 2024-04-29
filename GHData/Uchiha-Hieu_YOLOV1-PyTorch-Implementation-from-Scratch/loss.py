import torch
import numpy as np
import torch.nn.functional as F
from utils import cal_iou
from utils import normalize_pred_boxes,truebox_to_cell

def YoloLoss(preds,targets,cell_size=7,num_boxes=2,num_classes=3,device="cpu"):
    """
    preds : predicted label matrix with shape (batchsize,cell_size,cell_sizel,5*num_boxes+C)
        # self.B*5 + self.C : 
        # B first elements are confident score of each box
        # 4*B next elements is coordinates of each box (box1_x,box1_y,box1_w,box1_h,box2_x,box2_y,box2_w,box2_h,....)]
        # C last elements : predict probability of class for boxes of each cell
        coordinates of pred boxes are calculated on cell coordinate and cellsize,so when calculating loss ->normalize to image size

    targets : ground truth label matrix with shape (batchsize,cell_size,cell_size,5+C)
        # 5+C:
        # 5 : confident score,x,y,w,h
        # C : predict probability of class for each cell
    
    return : loss value
    """
    
    #get coordinates:
    pred_boxes = preds[:,:,:,num_boxes:5*num_boxes]
    pred_cell_boxes = pred_boxes.view(pred_boxes.shape[0],pred_boxes.shape[1],pred_boxes.shape[2],num_boxes,4) #batchsize,cell_size,cell_size,num_boxes,4
    pred_true_boxes = normalize_pred_boxes(pred_cell_boxes,cell_size,num_boxes,device)
    gt_boxes = targets[:,:,:,1:5]
    gt_boxes_reshape = gt_boxes.view(gt_boxes.shape[0],gt_boxes.shape[1],gt_boxes.shape[1],1,4) #batchsize,cell_size,cell_size,1,4
    gt_true_boxes = torch.tile(gt_boxes_reshape,(1,1,1,num_boxes,1))/224. #batchsize,cell_size,cell_size,num_boxes,4
    gt_cell_boxes = truebox_to_cell(gt_true_boxes,cell_size,num_boxes,device) #batchsize,cell_size,cell_size,num_boxes,4
    
    #get classes
    pred_classes = preds[:,:,:,5*num_boxes:] #batchsize,cell_size,cell_size,num_classes
    gt_classes = targets[:,:,:,5:] #batchsize,cell_size,cell_size,num_classes
    
    #get confident scores
    gt_conf = targets[:,:,:,:1]
    pred_conf = preds[:,:,:,:num_boxes]

    #Calculate iou
    ious = cal_iou(pred_true_boxes,gt_true_boxes) #batchsize,cell_size,cell_size,num_boxes
    obj_mask = torch.max(ious,dim=3,keepdim=True)[0]
    iou_metric = torch.mean(torch.sum(obj_mask,dim=[1,2,3])/torch.sum(gt_conf,axis=[1,2,3]))
    obj_mask_final = (ious >= obj_mask).type(torch.float32)*gt_conf
    noobj_mask = 1 - obj_mask_final
    
    #Get CLASS LOSS
    # class_loss = torch.mean(torch.sum((gt_conf*pred_classes-gt_conf*gt_classes)**2,dim=[1,2,3]))
    class_loss = F.mse_loss(gt_conf*pred_classes,gt_conf*gt_classes,reduction='sum')

    # #Get Object loss
    # obj_loss = torch.mean(torch.sum((obj_mask_final*pred_conf-1)**2,dim=[1,2,3]))
    obj_loss = F.mse_loss(obj_mask_final*pred_conf,ious,reduction='sum')

    # #Get no Object loss
    # noobj_loss = torch.mean(torch.sum((noobj_mask*pred_conf)**2,dim=[1,2,3]))
    noobj_loss = F.mse_loss(noobj_mask*pred_conf,torch.zeros_like(pred_conf),reduction='sum')

    # #box loss
    box_mask = obj_mask_final.unsqueeze(-1)
    # box_loss = torch.mean(torch.sum((box_mask*pred_cell_boxes-box_mask*gt_cell_boxes)**2,dim=[1,2,3]))
    box_loss = F.mse_loss(box_mask*pred_cell_boxes,box_mask*gt_cell_boxes,reduction='sum')

    loss = class_loss + obj_loss + 0.5*noobj_loss + 5*box_loss
    return loss/pred_boxes.shape[0],class_loss/pred_boxes.shape[0],(obj_loss + 0.5*noobj_loss)/pred_boxes.shape[0],5*box_loss/pred_boxes.shape[0],iou_metric
