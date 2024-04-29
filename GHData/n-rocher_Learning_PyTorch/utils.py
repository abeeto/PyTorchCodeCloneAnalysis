import torch
import wandb
import numpy as np
import torch.nn as nn

class ArgmaxIOU(nn.Module):
    def __init__(self, n_class):
        super(ArgmaxIOU, self).__init__()
        self.n_class = n_class

    def iou(self, y_pred, y_true):

        y_true = torch.argmax(y_true, dim=0)
        y_pred = torch.argmax(y_pred, dim=0)

        IOU = []
        for c in range(self.n_class):

            TP = torch.sum((y_true == c) & (y_pred == c)).item()
            FP = torch.sum((y_true != c) & (y_pred == c)).item()
            FN = torch.sum((y_true == c) & (y_pred != c)).item()

            n = TP
            d = TP + FN + FP

            if TP != 0 and d != 0:
                iou = torch.div(n, d)
                IOU.append(iou)

        return torch.mean(torch.FloatTensor(IOU))

    def forward(self, prediction, target):
        batch = prediction.shape[0]

        score = []
        for idx in range(batch):
            iou_value = self.iou(prediction[idx], target[idx])
            score.append(iou_value)

        return torch.mean(torch.FloatTensor(score))


def convertPredictionsForWB(data, class_labels):
    result = []
   
    for i in range(len(data)):
        img_pred = wandb.Image(np.moveaxis(data[i][0].cpu().numpy(), 0, 2), masks={
            "predictions": {
                "mask_data": torch.argmax(data[i][1], dim=0).cpu().numpy(),
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": torch.argmax(data[i][2], dim=0).cpu().numpy(),
                "class_labels": class_labels
            }
        })

        result.append(img_pred)
    
    return result

def save_model(filename, model, optimizer, epoch, config, dataset):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'dataset': dataset,
    }, filename)