import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
        This function transform the feature obtained at the end of the network 
        into an array where each row is a detection
    """
    
    batch_size = prediction.size(0)
    
    stride =  inp_dim // prediction.size(2)     # the total stride of the network (from input to prediction)
    
    grid_size = prediction.size(2) #inp_dim // stride
    bbox_attrs = 5 + num_classes
    nb_anchors = len(anchors)
    
    # rescale the anchor to the output map size
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # B x C x H x W -> B x C x HW
    prediction = prediction.view(batch_size, bbox_attrs*nb_anchors, grid_size*grid_size)
    # B x C x HW -> B x HW x C
    prediction = prediction.transpose(1,2).contiguous()
    # B x HW x C -> B x HW*nb_anchors x bbox_attrs (a line contained 
    # the bbox_attrs for each anchors are split on multiple lines 
    # so that one bbox_attrs per line)
    prediction = prediction.view(batch_size, grid_size*grid_size*nb_anchors, bbox_attrs)


    # Sigmoid the position and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    
    
    # Add the center offsets for each cell of the grid
    y, x = np.mgrid[:grid_size, :grid_size]
    x_offset = torch.FloatTensor(x.repeat(nb_anchors))
    y_offset = torch.FloatTensor(y.repeat(nb_anchors))
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    prediction[:, :, 0] += x_offset.unsqueeze(0)
    prediction[:, :, 1] += y_offset.unsqueeze(0)
    
    # Log space transform height and the width with the anchors dimensions
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    #Softmax the class scores
    prediction[:, :, 5:] = torch.sigmoid(prediction[:, :, 5:])

    # rescale position and dimension to the input images scale
    prediction[:, :, :4] *= stride
   
    return prediction

    
def filter_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    """ 
        Filter the result with non max suppression
    """
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    

    # transform x, y, w, h -> x1, y1, x2, y2
    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2) 
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_a[:, :, :4]

    
    batch_size = prediction.size(0)
    
    
    # group every true detection in a table with a batch id as first column
    output = []
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
                
        # Get the class having maximum score and the index of that class
        # Get rid of num_classes softmax scores 
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_index = torch.max(image_pred[:, 5:5 + num_classes], axis=1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_index)
        image_pred = torch.cat(seq, 1)
        
        
        #Get rid of the zero entries
        non_zero_ind =  torch.nonzero(image_pred[:, 4])
        image_pred_ = image_pred[non_zero_ind.squeeze(), :]
        
        if image_pred_.dim() == 1:
            image_pred_ = image_pred_.unsqueeze(0)
        
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])
        
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            image_pred_class = image_pred_[image_pred_[:, -1] == cls, :]    
            
            # Sort the detections by confidence
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index, :]
            
            #if nms has to be done
            if nms:
                # Keep only the real maxima
                to_keep, non_max = [], []
                for i in range(image_pred_class.size(0)):
                    # Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    if i not in non_max:
                        to_keep.append(i)
                        if i < image_pred_class.size(0) - 1:
                            ious = bbox_IoU(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                            similar_bbox = torch.where(ious > nms_conf)[0]
                            similar_bbox += i+1     # shift to get the real indices
                            non_max.extend(similar_bbox.tolist())
                image_pred_class = image_pred_class[to_keep, :]
                    
                    
            # add batch index as first column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            true_det_with_ind = torch.cat((batch_ind, image_pred_class), 1)
            output.append(true_det_with_ind)
    if len(output) > 0:
        return torch.cat(output)
    else:
        return torch.Tensor([])


    
def bbox_IoU(box1, box2):
    """
        Compute the inserction over union of box1 with all the box2
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # area of intersection            
    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)
    
    # area of union
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou
    
    
                        
        
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    maxi_scale = min(w/img_w, h/img_h)
    new_w = int(img_w * maxi_scale)
    new_h = int(img_h * maxi_scale)
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    # put it into an empty image of the desired size
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,
           (w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas


def prepare_image(img, inp_dim):
    """
        Prepare an image for inputting the network
    """
    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    
    img = letterbox_image(orig_im, (inp_dim, inp_dim))
    # change BGR -> RGB and HxWxC -> CxHxW
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy() 
    # add a dimension at the front
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img, orig_im, dim
