import torch
import numpy as np

def create_idx_matrix(cell_size=7,num_boxes=2,device="cpu"):
    cell_idxs = torch.tensor([np.arange(cell_size)]*(cell_size*num_boxes))
    cell_idxs = cell_idxs.view(num_boxes,cell_size,cell_size)
    cell_idxs = cell_idxs.permute(1,2,0)
    cell_idxs = cell_idxs[None,:]
    cell_idxs_tran = cell_idxs.permute(0,2,1,3)
    return cell_idxs.to(device),cell_idxs_tran.to(device)

def cal_iou(pred_boxes,target_boxes):
    """
    pred_boxes : tensor with shape (batchsize,cell_size,cell_size,num_boxes,4)
    target_boxes : tensor with shape (batchsize,cell_size,cell_size,num_boxes,4)
    (target_boxes should be cast shape as pred_boxes by torch.tile because num_boxes of target is 1)
    return iou : tensor with shape (batchsize,cell_size,cell_size,num_boxes)
    """
    pred_x1 = pred_boxes[:,:,:,:,0]-pred_boxes[:,:,:,:,2]/2.
    pred_x2 = pred_boxes[:,:,:,:,0]+pred_boxes[:,:,:,:,2]/2.
    pred_y1 = pred_boxes[:,:,:,:,1]-pred_boxes[:,:,:,:,3]/2.
    pred_y2 = pred_boxes[:,:,:,:,1]+pred_boxes[:,:,:,:,3]/2.

    target_x1 = target_boxes[:,:,:,:,0]-target_boxes[:,:,:,:,2]/2.
    target_x2 = target_boxes[:,:,:,:,0]+target_boxes[:,:,:,:,2]/2.
    target_y1 = target_boxes[:,:,:,:,1]-target_boxes[:,:,:,:,3]/2.
    target_y2 = target_boxes[:,:,:,:,1]+target_boxes[:,:,:,:,3]/2.

    xmin = torch.max(pred_x1,target_x1)
    ymin = torch.max(pred_y1,target_y1)
    xmax = torch.min(pred_x2,target_x2)
    ymax = torch.min(pred_y2,target_y2)

    inter = torch.clamp(xmax-xmin,0)*torch.clamp(ymax-ymin,0)
    area1 = torch.clamp(pred_x2-pred_x1,0)*torch.clamp(pred_y2-pred_y1,0)
    area2 = torch.clamp(target_x2-target_x1,0)*torch.clamp(target_y2-target_y1,0)
    return inter/(area1+area2-inter+1e-6)

def normalize_pred_boxes(pred_boxes,cell_size=7,num_boxes=2,device="cpu"):
    """
    pred_boxes : tensor with shape (batchsize,cell_size,cell_size,num_boxes,4)
    return pred_boxes with normalized type based on image size 
    x_cell -> x_img (range 0->1)
    y_cell -> y_img (range 0->1)
    ...
    """
    cell_idxs,cell_idxs_tran = create_idx_matrix(cell_size,num_boxes,device)
    new_x = ((pred_boxes[:,:,:,:,0]+cell_idxs)/cell_size).unsqueeze(-1)
    new_y = ((pred_boxes[:,:,:,:,1]+cell_idxs_tran)/cell_size).unsqueeze(-1)
    new_w = ((pred_boxes[:,:,:,:,2])**2).unsqueeze(-1)
    new_h = ((pred_boxes[:,:,:,:,3])**2).unsqueeze(-1)
    return torch.cat((new_x,new_y,new_w,new_h),dim=4)

def truebox_to_cell(gt_boxes,cell_size=7,num_boxes=2,device="cpu"):
    """
    gt_boxes : tensor with shape (batchsize,cell_size,cell_size,num_boxes,4)
    return gt_boxes in cell coordinates and cell cell_size
    """
    cell_idxs,cell_idxs_tran = create_idx_matrix(cell_size,num_boxes,device)
    new_x = (gt_boxes[:,:,:,:,0]*cell_size-cell_idxs).unsqueeze(-1)
    new_y = (gt_boxes[:,:,:,:,1]*cell_size-cell_idxs_tran).unsqueeze(-1)
    new_w = (torch.sqrt(gt_boxes[:,:,:,:,2])).unsqueeze(-1)
    new_h = (torch.sqrt(gt_boxes[:,:,:,:,3])).unsqueeze(-1)
    # gt_boxes[:,:,:,:,0] = gt_boxes[:,:,:,:,0]*cell_size-cell_idxs #x_cell
    # gt_boxes[:,:,:,:,1] = gt_boxes[:,:,:,:,1]*cell_size-cell_idxs_tran #y_cell
    # gt_boxes[:,:,:,:,2] = torch.sqrt(gt_boxes[:,:,:,:,2]) #w_cell
    # gt_boxes[:,:,:,:,3] = torch.sqrt(gt_boxes[:,:,:,:,3]) #h_cell
    return torch.cat((new_x,new_y,new_w,new_h),dim=4)

# x = torch.randn(64,7,7,2,4)
# y = torch.randn(64,7,7,2,4)
# print(cal_iou(x,y).shape)