import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable


def bce_with_logits(logits, gts):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, gts)
    return loss


def train(model, train_loader, eval_loader, save_path, args):
    num_epoch = args.epoch
    utils.create_dir(save_path)
    optim = torch.optim.Adamax(model.parameters(), lr=args.lr)
    logger = utils.Logger(os.path.join(save_path, 'log.txt'))
    
    best_eval_loss = 1000000.00
    best_eval_roc = 0.0
    if os.path.exists(os.path.join(save_path, 'model.pth')):
        checkpoint = torch.load(os.path.join(save_path, 'model.pth'))
        model.load_state_dict(checkpoint)
        print('[*] Saved model is loaded:\t', save_path+'/model.pth')
    
    for epoch in range(num_epoch):
        total_loss = 0
        t = time.time()

        for idx, (vital, gt) in enumerate(train_loader):
            vital = Variable(vital.float()).cuda()
            #demo = Variable(demo.float()).cuda()
            gt = Variable(gt.float()).cuda()

            pred = model(vital)
            #loss = bce_with_logits(pred, gt)
            loss = F.mse_loss(pred, gt)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_loss += loss.data

        total_loss /= len(train_loader.dataset)
        model.train(False)
        eval_loss, _ , eval_roc = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.5f' % (total_loss))
        
        print('epoch %d, time: %.2f' %(epoch, time.time()-t))
        print('\ttrain_loss: %.8f, \teval_loss: %.8f' %(total_loss, eval_loss))
        #if eval_loss < best_eval_loss:
        if eval_roc > best_eval_roc:
            model_path = os.path.join(save_path, 'model.pth')
            torch.save(model.state_dict(), model_path)
            #best_eval_loss = eval_loss
            best_eval_roc = eval_roc

def evaluate(model, dataloader, reload=False, save_path=None):
    if reload:
        try:
            checkpoint = torch.load(os.path.join(save_path, 'model.pth'))
            model.load_state_dict(checkpoint)
            print('[*] Saved model is loaded:\t', save_path+'/model.pth')
        except:
            raise 

    loss = 0
    num_data = 0
    preds = []
    gts = []
    for vital,  gt in iter(dataloader):
        gts.extend(gt.data.numpy())
        vital = Variable(vital.float(), volatile=True).cuda() 
        #demo = Variable(demo.float(), volatile=True).cuda() 
        gt = Variable(gt.float(), volatile=True).cuda()         
        pred = model(vital)
        preds.extend(pred.data.cpu().numpy())
        loss += bce_with_logits(pred, gt)
        
        num_data += pred.size(0)
    
    roc = utils.get_roc_score(gts, np.array(preds), 0.3)
    print('Pearson:\t', utils.get_pearson(gts, np.array(preds))[0])
    print('ROC:\t', roc)
    return loss / num_data, np.array(preds), roc 
