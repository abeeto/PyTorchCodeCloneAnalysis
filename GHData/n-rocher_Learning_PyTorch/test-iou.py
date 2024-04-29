from dataloader import A2D2_Dataset
import cv2
import numpy as np
from tqdm import tqdm
import random

def iou(target, prediction):
    ious = []
    print(target.shape)
    target = target.view(-1)
    print(target.shape)
    prediction = prediction.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, len(target)):  # This goes from 1:n_class-1 -> class "0" is ignored
        pred_inds = prediction == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    return np.array(ious)


if __name__ == "__main__":

    training_dataset = A2D2_Dataset("training", size=(512, 400))

    for id in range(2000):
        img, target = training_dataset.__getitem__(id)

        print(iou(target, target))