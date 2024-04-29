import torchmetrics
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np

#-------------- Usando las Implementaciones de torchmetrics -----------------#
def tm(preds,target, n_clases= None, mode = 'normal', mdmc = None, prnt = None):
    if mode == 'normal':
        if len(preds.shape) == 1:
            accuracy = torchmetrics.Accuracy()
            precision = torchmetrics.Precision()
            recall = torchmetrics.Recall()
            f1score = torchmetrics.F1Score()
        elif len(preds.shape) == 2:
            accuracy = torchmetrics.Accuracy(mdmc_reduce = mdmc)
            precision = torchmetrics.Precision(mdmc_average = mdmc)
            recall = torchmetrics.Recall(mdmc_average = mdmc)
            f1score = torchmetrics.F1Score(mdmc_average = mdmc)
    else:
        accuracy = torchmetrics.Accuracy(num_classes=n_clases, average = mode, mdmc_average = mdmc)
        precision = torchmetrics.Precision(num_classes=n_clases, average = mode, mdmc_average = mdmc)
        recall = torchmetrics.Recall(num_classes=n_clases, average = mode, mdmc_average = mdmc)
        f1score = torchmetrics.F1Score(num_classes=n_clases, average = mode, mdmc_average = mdmc)
        if (mode is 'none' or mode is None) and prnt is True :
            print('Mode {}: \n Acc:{} - Pres:{} - Rec:{} - F1Sc:{}'.format(mode,accuracy(preds, target),precision(preds, target),recall(preds, target),f1score(preds, target)))
            return accuracy(preds, target),precision(preds, target),recall(preds, target),f1score(preds, target)

    if prnt is True:
        print('Mode {}: \n Acc:{:.4f} - Pres:{:.4f} - Rec:{:.4f} - F1Sc:{:.4f}'.format(mode,accuracy(preds, target),precision(preds, target),recall(preds, target),f1score(preds, target)))

    return accuracy(preds, target),precision(preds, target),recall(preds, target),f1score(preds, target)

#-------------- Usando las Implementaciones de sklearn -----------------#

def sklearn_metrics(pred, target, threshold=0.5):
  pred = np.array(pred > threshold, dtype=float)
  # print('Pred:', pred, 'Target: ', target)
  return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division = 0),
          'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division = 0),
          'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division = 0),
          'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division = 0),
          'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division = 0),
          'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division = 0),
          'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples', zero_division = 0),
          'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples', zero_division = 0),
          'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples', zero_division = 0),
          }

#-------------- Implementaciones propias de las métricas -----------------#

def vals(preds, target, mode = 'normal', cls = None):
    tp,tn,fp,fn = 0,0,0,0
    if mode == 'normal':
        for i in range(len(preds)):
          if preds[i] == target[i]:
              tp += 1
              tn += 1
          else:
              fp += 1
              fn += 1
    elif mode in ['macro', 'weighted','none']:
        target = target.numpy()
        preds = preds.numpy()
        dels = []
        for i in np.argwhere(target == cls):
            if preds[i] == cls:
                tp += 1
                dels.append(i)
            else:
                fn += 1
                dels.append(i)
        target = np.delete(target,dels)
        preds = np.delete(preds,dels)

        fps = np.argwhere(preds == cls)
        fp =  len(fps)
        if fp != 0:
            target = np.delete(target,fp)
            preds = np.delete(preds,fp)

        tn = len(target)
    elif mode == 'samples':
        for i in range(len(preds)):
            if preds[i] == 1 and preds [i] == target[i]:
                tp += 1
            elif preds[i] == 0 and preds [i] == target[i]:
                tn += 1
            elif preds[i] == 1 and preds [i] != target[i]:
                fp += 1
            elif preds[i] == 0 and preds [i] != target[i]:
                fn +=1

    return tp,tn,fp,fn

def imp_metrics(preds,target, n_clases = None, mode = 'normal', prnt = None):
    if mode == 'normal':
        tp,tn,fp,fn = vals(preds, target)
        acc = (tp+tn)/(tp+tn+fp+fn)
        pres = (tp)/(tp+fp)
        rec = (tp)/(tp+fn)
        f1sc = 2*((pres*rec)/(pres+rec))
    elif mode == 'micro':
        tp,tn,fp,fn = vals(preds, target)
        acc = (tp+tn)/(tp+tn+fp+fn)
        pres = (tp)/(tp+fp)
        rec = (tp)/(tp+fn)
        f1sc = 2*((pres*rec)/(pres+rec))
    elif mode == 'macro':
        acc,pres,rec,f1sc = 0,0,0,0
        for i in range(n_clases):
            tp,tn,fp,fn = vals(preds, target,mode,i)
            acc += (tp+tn)/(tp+tn+fp+fn)
            pres += (tp)/(tp+fp)
            rec += (tp)/(tp+fn)
            if pres != 0:
                f1sc += 2*((pres*rec)/(pres+rec))
            else:
                f1sc += 0
        acc,pres,rec,f1sc = [(1/n_clases)*i for i in [acc,pres,rec,f1sc]]
        # f1sc += 2*((pres*rec)/(pres+rec))
    elif mode == 'weighted':
        acc,pres,rec,f1sc = 0,0,0,0
        for i in range(n_clases):
            tp,tn,fp,fn = vals(preds, target,mode,i)
            acc += ((tp+fn)/(tp+tn+fp+fn))*(tp+tn)/(tp+tn+fp+fn)
            pres += ((tp+fn)/(tp+tn+fp+fn))*(tp)/(tp+fp)
            rec += ((tp+fn)/(tp+tn+fp+fn))*(tp)/(tp+fn)
            if pres != 0:
                f1sc += 2*((pres*rec)/(pres+rec))
            else:
                f1sc += 0
    elif mode == 'none':
        acc,pres,rec,f1sc = [],[],[],[]
        for i in range(n_clases):
            tp,tn,fp,fn = vals(preds, target,mode,i)
            acc.append((tp+tn)/(tp+tn+fp+fn))
            pres.append((tp)/(tp+fp))
            rec.append((tp)/(tp+fn))
            if pres[-1] != 0:
                f1sc.append(2*((pres[-1]*rec[-1])/(pres[-1]+rec[-1])))
            else:
                f1sc.append(0)
        if prnt is True:
            print('Mode {}: \n Acc:{} - Pres:{} - Rec:{} - F1Sc:{}'.format(mode,acc,pres,rec,f1sc))
        return acc, pres, rec, f1sc
    elif mode == 'samples':
        acc,pres,rec,f1sc = 0,0,0,0
        for i in range(len(preds)):
            tp,tn,fp,fn = vals(preds[i], target[i], mode)
            acc += (tp+tn)/(tp+tn+fp+fn)
            pres += (tp)/(tp+fp)
            rec += (tp)/(tp+fn)
            # if pres != 0:
            #   f1sc += 2*((pres*rec)/(pres+rec))
            # else:
            #   f1sc += 0
        f1sc += 2*((pres*rec)/(pres+rec))
        acc,pres,rec,f1sc = [(1/len(preds))*i for i in [acc,pres,rec,f1sc]]

    if prnt is True:
        print('Mode {}: \n Acc:{:.4f} - Pres:{:.4f} - Rec:{:.4f} - F1Sc:{:.4f}'.format(mode,acc,pres,rec,f1sc))

    return acc, pres, rec, f1sc
