from genericpath import exists
from tracemalloc import start
import torch
import torch.nn as nn
from utils import intersection_over_union

""" 
The loss function is based on the paper: You Only Look Once: Unified, Real-Time Object Detection
by Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

The idea is that the YOLO predicts multiple bounding boxes per grid cell (in our case 2).

During training we only want box predictor to be responsible for each object, we decide that
by taking the highest current IOU (intersection over union) with the ground truth (the given box).
We can achieve this via an identity function (Iobj_ij).

Iobj_i is 1 if an object appears in cell i, 0 otherwise
Iobj_ij is 1 if Iobj_i is 1 AND the jth box in cell i is responsible for the object.

The loss function is composed of 4 parts:
    It takes the sum squared error of the midpoints of the ground truth and the predicted (with prio lambda coord)
    Plus the sum squared error of the square rooted heights and widths (with prio lambda coord)
    Plus the probability of there being an object in the cell
    Plus the probability that there is no object in the cell (accomplished by noobj identity function, 
                                                              works the same as the obj identity functions
                                                              but negated with prio lambfa NOOBJ)
    Plus the sum squared error of the classes (deviation from the target class)

These make it so that we use a regression-based loss on everything, which simplifies the loss function a bit

See loss.png
"""

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20): # for YOLO on PASCAL VOC, the paper uses S = 7 (so breaks the image into 7x7 grid) and B = 2 (two bounding boxes)
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lamdba_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5) # tensor structure according to the paper (SxSx(B*5+C)) so the final prediction is a 7 x 7 x 30 tensor
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0) # best_box returns the argmax for the iou_maxes
        box_exists = target[..., 20].unsqueeze(3) # Iobj_i in paper

        # BOX COORDINATES

        box_predictions = box_exists * (
            ( 
                best_box * predictions[..., 26:30] +
                (1 - best_box) * predictions[..., 21:25]
            )
        )

        box_targets = box_exists * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S,4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # OBJECT LOSS

        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(box_exists * pred_box),
            torch.flatten(box_exists * target[..., 20:21])
        )

        # NO OBJECT LOSS

        no_object_loss = self.mse(
            torch.flatten((1 - box_exists) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - box_exists) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - box_exists) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - box_exists) * target[..., 20:21], start_dim=1)
        )

        # CLASS LOSS

        class_loss = self.mse(
            torch.flatten(box_exists * predictions[..., :20], end_dim=-2),
            torch.flatten(box_exists * target[..., :20], end_dim=-2)
        )

        #ACTUAL LOSS

        loss = (
            self.lambda_coord * box_loss # First two rows of loss in paper
            + object_loss
            + self.lamdba_noobj * no_object_loss
            + class_loss
        )

        return loss




