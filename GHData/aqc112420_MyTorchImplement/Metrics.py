from sklearn.metrics import confusion_matrix
import numpy as np

class Iou(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def get_confusion_matrix(self, y_true, y_pred):
        try:
            y_true_flatten = y_true.numpy().flatten()
            y_pred_flatten = y_pred.numpy().argmax(axis=1).flatten()
        except TypeError as e:
            y_true_flatten = y_true.cpu().numpy().flatten()
            y_pred_flatten = y_pred.cpu().numpy().argmax(axis=1).flatten()
        confusion_mat = confusion_matrix(y_true_flatten, y_pred_flatten)
        return confusion_mat

    def IOU(self, y_true, y_pred):
        mat = self.get_confusion_matrix(y_true, y_pred)
        IoU = np.diag(mat) / (
                np.sum(mat, axis=1) + np.sum(mat, axis=0) -
                np.diag(mat))
        MIoU = np.nanmean(IoU)  # 跳过0值求mean,shape:[21]
        return IoU, MIoU