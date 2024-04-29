import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    # TODO: grid_size ==prediction.size(2)?
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    # TODO: need to including the last one
    prediction[:, :, 5:] = torch.sigmoid(prediction[:, :, 5:])

    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    return: IoU of two boxes
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # TODO: WHY +1
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1+1,
                             min=0) * torch.clamp(inter_rect_y2-inter_rect_y1+1, min=0)

    #TODO: WHY +1
    b1_area = (b1_x2-b1_x1+1) * (b1_y2-b1_y1+1)
    b2_area = (b2_x2-b2_x1+1) * (b2_y2-b2_y1+1)

    iou = inter_area / (b1_area+b2_area-inter_area)
    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # whether there is an object
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    # Non-maximun Suppression
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False
    for index in range(batch_size):
        image_pred = prediction[index]
        max_conf, max_conf_score = torch.max(image_pred[:, 5:], 1)
        max_conf = max_conf.float().unsqueeze(1)

        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        # print(image_pred[:,:5].size(), max_conf.size(), max_conf_score.size())
        image_pred = torch.cat(seq, 1)

        non_zero_index = torch.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_index.squeeze(), :].view(-1, 7)
        except:
            continue
        if image_pred_.shape[0] == 0:
            continue
        img_classes = unique(image_pred_[:, -1])
        for cls in img_classes:
            cla_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_index = torch.nonzero(cla_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_index].view(-1, 7)

            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                        0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                non_zero_index = torch.nonzero(
                    image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_index].view(-1, 7)
            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(index)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w) //
           2:(w-new_w)//2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 

    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def test_read_imgs(folder):
    """read imgs from folder: imgs

    return:
        img_list: list contains img name
        img_data: list contains img data(3 * 608 * 608), rgb
        img_size: list contains img width and height
    """
    img_list = os.listdir(folder)
    img_size = []
    img_data = []
    for i in range(len(img_list)):
        img = cv2.imread(os.path.join(folder, img_list[i]) )
        img_size.append(img.shape[:2])
        img_data.append(img)
    return img_list, img_data, img_size


def test_save_img(name, img_data, img_size, prediction, objs):
    """draw box on img and save to disk
    Arguments: 
        name: img name
        img_data: img metadata, the output of cv2.imread
        img_size: tuple (height, width)
        prediction: prediction output (n*8), n means the num of objects in one image
        objs: total objects name in this image, list
    """
    num = 0
    height, width = img_size
    for one_box in prediction:
        box = one_box[1:5]
        x1 = box[0] * width / 608
        y1 = box[1] * height / 608
        x2 = box[2] * width / 608
        y2 = box[3] * height / 608

        obj = objs[num]
        num += 1

        cv2.rectangle(img_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_data, obj, (x1, y1-10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(name, img_data)
