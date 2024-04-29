import torch
import torch.nn as nn
import cv2
import numpy as np

###############################################################################
# Preprocessing
###############################################################################

def preprocess_image(image):
    # Convert color space BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    # Compute new width and height
    dim = 416
    scale = dim / max(image.shape[:2])
    new_height = int(scale * height)
    new_width = int(scale * width)
    # Resize image
    image = cv2.resize(image, (new_width, new_height))
    # Pad image (Use padding to keep aspect ratio)
    top = bottom = (dim - new_height) // 2
    left = right = (dim - new_width) // 2
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    # Reshape tensor HxWxC to CxHxW
    image = image.transpose(2, 0, 1)
    # Add extra dimension CxHxW to NxCxHxW
    image = image[np.newaxis, ...]
    # Normalise image
    image = image / 255.0
    # Convert to tensor
    return torch.from_numpy(image).float()


###############################################################################
# Post-processing
###############################################################################
def IOU(box, boxes):
    # Get coordinates of boxes
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # Get intersection coordinates
    intersect_x1 = torch.max(x1, x1s)
    intersect_y1 = torch.max(y1, y1s)
    intersect_x2 = torch.min(x2, x2s)
    intersect_y2 = torch.min(y2, y2s)
    # Calculate intersection area
    intersect_width = torch.clamp(intersect_x2 - intersect_x1 + 1, min=0)
    intersect_height = torch.clamp(intersect_y2 - intersect_y1 + 1, min=0)
    intersect_area = intersect_width * intersect_height
    # Calculate union area
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxes_area = (x2s - x1s + 1) * (y2s - y1s + 1)
    union_area = box_area + box_area - intersect_area
    # Calculate IOU
    return intersect_area / (union_area + 1e-7)

def NMS(predictions):
    # Convert bounding box attributes from (x,y,w,h) to (x1, y1, x2, y2)
    corners = predictions[..., :4].new(predictions[..., :4].shape)
    corners[..., 0] = predictions[..., 0] - predictions[..., 2] / 2
    corners[..., 1] = predictions[..., 1] - predictions[..., 3] / 2
    corners[..., 2] = predictions[..., 0] + predictions[..., 2] / 2
    corners[..., 3] = predictions[..., 1] + predictions[..., 3] / 2
    predictions[..., :4] = corners

    output = [None in range(len(predictions))]
    # Iterate through each image
    for i, prediction in enumerate(predictions):
        # Filter predictions with object score below threshold
        prediction = prediction[prediction[..., 4] > 0.6]
        # Get class with the highest confidence
        confidence, labels = torch.max(prediction[..., 5:], 1, keepdim=True)
        prediction = torch.cat((prediction[..., :5], confidence, labels), 1)
        # For each class
        boxes = []
        for label in torch.unique(labels):
            # Get all predictions from the same class
            class_predictions = prediction[prediction[..., -1] == label]
            # Sort predictions by object confidence
            indices = torch.sort(class_predictions[..., 4], descending=True)[1]
            class_predictions = class_predictions[indices]
            # Iterate througth each detection
            while class_predictions.size(0):
                # Compute Intersection over Union
                ious = IOU(class_predictions[0, :4].unsqueeze(0), class_predictions[:, :4])
                # Remove all predictions with IOU larger than threshold
                repeated = ious > 0.4
                # Merge bouding boxes by weight
                weights = class_predictions[repeated, 4].unsqueeze(1)
                class_predictions[0, :4] = (weights * class_predictions[repeated, :4]).sum(0) / weights.sum()
                # Add box with the hightest object score (Already sorted)
                boxes.append(class_predictions[0])
                # Remove repeated boxes with high overlapping area
                class_predictions = class_predictions[~repeated]
        output[i] = torch.stack(boxes)
    return output

###############################################################################
# Bouding Boxes
###############################################################################

def rescale(boxes, dimensions):
    dim = 416
    height, width = dimensions
    scale = dim / max(dimensions)
    # Remove left padding if exists
    boxes[:,[0,2]] -= (dim - scale * width) / 2
    # Remove top padding if exists
    boxes[:,[1,3]] -= (dim - scale * height) / 2
    # Scale up image to original size
    boxes[:, :4] /= scale
    return boxes
