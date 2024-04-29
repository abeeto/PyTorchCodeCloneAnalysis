import numpy as np
import cv2

def _fast_hist(label_true, label_pred, n_class=2):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class=2):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten())
    acc = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return acc, mean_iu

def background_subtraction(img, pred):
    return (img * pred).transpose(1, 2, 0)

def noise_reduction(img):
    img[img==1] = 255
    #ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, 9)
    kernel = np.ones((9,9), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #img = cv2.dilate(img, kernel)
    binary, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    if len(area) == 1:
        img[img==255] = 1
        return np.expand_dims(img, axis=0)
    else:
        max_idx = np.argmax(area)
        cv2.fillPoly(img, [contours[max_idx]], 128)
        img[img==255] = 0
        img[img==128] = 1
        return np.expand_dims(img, axis=0)
