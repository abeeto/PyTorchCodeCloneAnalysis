import torch
import shutil
import numpy as np
from config import configDict
from visdom import Visdom

import matplotlib.pyplot as plt
import imshowpair

SMOOTH = 1e-6

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    
    torch.save(state, checkpoint_dir + 'checkpoint.pt')
    if is_best:
        best_fpath = best_model_dir +'/best_model.pt'
        shutil.copyfile(checkpoint_dir + 'checkpoint.pt', best_fpath)


def load_ckp(checkpoint_fpath, encoder, student, teacher = None, optimizer = None):
    checkpoint = torch.load(checkpoint_fpath)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    student.load_state_dict(checkpoint['student_state_dict'])
    if teacher is not None and optimizer is None:
        teacher.load_state_dict(checkpoint['teacher_state_dict'])
        return encoder, student, teacher, checkpoint['epoch']
    if optimizer is not None and teacher is not None: 
        optimizer.load_state_dict(checkpoint['optimizer'])
        return encoder, student, teacher, optimizer, checkpoint['epoch']
    if optimizer is not None and teacher is None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return encoder, student, optimizer, checkpoint['epoch']
    return encoder, student, checkpoint['epoch']
    

def save_predictions(imgList, path):
    numImages = len(imgList)
    fig = plt.figure(figsize=(8,2))
    for i in range(0,numImages):
        plt.subplot(1,numImages, i+1)
        plt.imshow(imgList[i], aspect='auto')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def pixel_accuracy(output, target):
    accuracy = []
    N, _, h, w = output.shape
    output = output.data.cpu().numpy()
    pred = output.transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
    target = target.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
    for p,t in zip(pred, target):
        correct = (p == t).sum()
        total   = (t == t).sum()
        accuracy.append(correct/total)
    
    return accuracy

def f1(iou):
    dice = (2*iou)/(1+ iou)
    return dice


def iou(output, target):
    N, _, h, w = output.shape
    output = output.data.cpu().numpy()
    pred = output.transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
    predIU = np.zeros((pred.shape[0],configDict['num_classes'], h, w))
    for i in range(pred.shape[0]):
        for c in range(configDict['num_classes']):
            predIU[i][c][pred[i] == c] = 1
    
    predIU = predIU.astype(np.uint8)
    target = target.cpu().numpy()
    target = target.astype(np.uint8)
    batchIOU = (np.nanmean(classIU(predIU, target)))
    return batchIOU


def classIU(pred, target):
    N,c,_,_ = pred.shape
    iou = np.zeros((N,c))
    for i in range(N):
        for j,(p,t) in enumerate(zip(pred[i],target[i])):
            intersection = (p & t).sum()
            union = (p | t).sum()
            if intersection == 0 and union == 0:
                iou[i,j] = 'nan'
            else:
                iou[i,j]= ((intersection + SMOOTH)/(union + SMOOTH))
    return iou


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def getLabeledInBatch(target):

    target = target.detach().cpu().numpy()
    indices = []
    for i,t in enumerate(target):
        if -1 not in t:
            indices.append(i)
        else:
            continue
    return indices 

def getLabeledandWeaklyLabeled(target):
    target = target.detach().cpu().numpy()
    labeled_indices = []
    weakly_labeled_indices = []
    unlabeled_indices = []
    for i, t in enumerate(target):
        if t.shape == (configDict['num_classes'], 224, 224):
            labeled_indices.append(i)
        elif len(t.shape) == 2:
            weakly_labeled_indices.append(i)
        elif -1 in t:
            unlabeled_indices = []
    return labeled_indices, weakly_labeled_indices, unlabeled_indices

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    return sigmoid_rampup(epoch, 80)


def normalize(img):
    norm_img = (img - np.min(img))/np.ptp(img)
    norm_img[np.isnan(norm_img)] = 0
    return norm_img
