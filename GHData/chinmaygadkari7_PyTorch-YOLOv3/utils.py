import torch
import numpy as np
import random
import cv2

def calculate_area(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    area = width * height
    return area


def calculate_iou(one, rest):
    """
    Calculate IOU values for all remaining boxes against one box
    """
    # calculate area of boxes
    area_of_one = calculate_area(one[:, 0], one[:, 1], one[:, 2], one[:, 3])
    area_of_rest = calculate_area(rest[:, 0], rest[:, 1], rest[:, 2], rest[:, 3])

    # calculate intersection points for boxes
    intersect_x1 =  torch.max(one[:, 0], rest[:, 0])
    intersect_y1 =  torch.max(one[:, 1], rest[:, 1])
    intersect_x2 =  torch.min(one[:, 2], rest[:, 2])
    intersect_y2 =  torch.min(one[:, 3], rest[:, 3])

    # calculate area of intersection
    intersection_width = torch.max(intersect_x2 - intersect_x1).clamp(0)
    intersection_height = torch.max(intersect_y2 - intersect_y1).clamp(0)
    area_of_intersection = intersection_width * intersection_height

    iou = area_of_intersection / (area_of_one + area_of_rest - area_of_intersection)
    return iou


def non_maximum_suppression(prediction, nms_threshold=0.5):
    output = []
    class_ids = set(prediction[..., -1].tolist())
    for idx in class_ids:
        out = prediction[prediction[..., -1] == idx]
        sorted_idx = torch.sort(out[..., 4], descending = True )[1]
        out = out[sorted_idx]
        mask  = torch.ones(out.size(0))
        for id in range(mask.size(0) - 1):
            if not mask[id]:
                continue

            one = out[id].unsqueeze(0)
            rest = out[id + 1: ]
            iou = calculate_iou(one, rest)

            iou_mask = (iou < nms_threshold).float()
            mask[id + 1:] *= iou_mask

        out = out[torch.nonzero(mask)]
        output.append(out)

    output = torch.cat(output, 1).squeeze(0)
    return output


def xywh_to_xyxy(prediction):
    box = prediction.new(prediction.shape)
    box[...,0] = (prediction[...,0] - prediction[...,2] / 2)
    box[...,1] = (prediction[...,1] - prediction[...,3] / 2)
    box[...,2] = (prediction[...,0] + prediction[...,2] / 2)
    box[...,3] = (prediction[...,1] + prediction[...,3] / 2)
    prediction[...,:4] = box[...,:4]
    return prediction


def reduce_prediction(prediction, confidence=0.3, apply_nms=True, nms_threshold=0.5):
    # reduce datapoints to zero with objectness score < confidence
    objectness_scores = prediction[..., 4]
    mask = (objectness_scores > confidence).float().unsqueeze(-1)
    prediction *= mask

    non_zeros = torch.nonzero(prediction[..., 4]).transpose(-1, 0)[-1]
    if not torch.sum(non_zeros):
        return None

    prediction = prediction[0][non_zeros] # remove batch size as processing single example

    # convert predictions from xywh format to x1y1x2y2
    prediction = xywh_to_xyxy(prediction)

    # retrieve most confience classes and their scores
    max_confidence, max_confident_class = torch.max(prediction[..., 5:], 1)
    max_confidence = max_confidence.float().unsqueeze(1)
    max_confident_class = max_confident_class.float().unsqueeze(1)
    output = torch.cat((prediction[..., :5], max_confidence, max_confident_class), 1)

    if apply_nms:
        output = non_maximum_suppression(output, nms_threshold=nms_threshold)

    output = output.unsqueeze(0)
    return output

def scale_prediction(prediction, input_shape, target_shape):
    input_size = max(input_shape)
    target_height, target_width = target_shape[0], target_shape[1]

    scale_factor = min(input_size / target_height, input_size / target_width)
    x_displacement = (input_size - (scale_factor * target_width)) // 2
    y_displacement = (input_size - (scale_factor * target_height)) // 2
    prediction[..., [0, 2]] -= x_displacement
    prediction[..., [1, 3]] -= y_displacement
    prediction[..., :4]  = prediction[..., :4] / scale_factor

    return prediction

def make_color():
    color = tuple([random.randint(0, 255) for _ in range(3)])
    return color

def get_names(path):
    with open(path, 'r') as f:
        names = [
        l.strip() for l in f.readlines() if l.strip()
        ]
    return names

def plot_predictions(image, output):
    output = output.squeeze(0)
    predicted_classes = set(output[..., -1].tolist())
    color_code = {
        name: make_color() for name in predicted_classes
    }
    names = get_names('coco.names')
    for out in output:
        id = int(out[..., -1])
        color = color_code[id]
        name = names[id]
        bottom_left = tuple(out[..., [0, 1]].round().int())
        top_right = tuple(out[..., [2, 3]].round().int())
        cv2.rectangle(image, bottom_left, top_right, color, 2)
        cv2.putText(image, name, bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

    return image
