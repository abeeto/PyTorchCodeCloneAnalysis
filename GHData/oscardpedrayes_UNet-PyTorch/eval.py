import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix

from dice_loss import dice_coeff

def precision(outputs, labels, batchsize, calculate_cm):
    op = outputs
    la = labels.cpu().numpy().reshape(256*256*batchsize)
    _,preds= torch.max(op,dim=1)
    #print(la)
    preds= preds.cpu().numpy().reshape(256*256*batchsize)
    #print(preds)       
    #return precision_score(la,preds, average='macro', zero_division=0), recall_score(la,preds, average='macro', zero_division=0), accuracy_score(la,preds, normalize=True), confusion_matrix(la.view(), preds.view(), labels=[0,1,2,3])
    
    cm = None
    if calculate_cm:
        cm = confusion_matrix(la.view(), preds.view(), labels=[0,1,2,3])

    return accuracy_score(la,preds, normalize=True), cm 



def eval_net(net, loader, device, batchsize, calculate_cm=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of validation images
    tot = 0
    ga= 0.0
    cm = numpy.zeros(shape=(4,4), dtype=int)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type).squeeze(1)

            with torch.no_grad():
                mask_pred = net(imgs).squeeze(1)

            tot += F.cross_entropy(mask_pred, true_masks).item()
            tmp_ga, tmp_cm = precision(mask_pred, true_masks, len(imgs), calculate_cm)
            ga += tmp_ga
            if tmp_cm is None:
                cm = None
            else:
                cm += tmp_cm

            pbar.update()

    net.train()

    return tot / n_val, ga / n_val, cm

