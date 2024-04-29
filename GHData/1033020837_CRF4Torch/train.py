"""
训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils import data
from models import *
from utils import *
from config import *
from sklearn.metrics import *

# 日志模块
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 训练函数
def train(model, iterator, optimizer, epoch):
    model.train()
    losses = []  # 存储loss
    for i, batch in enumerate(iterator):
        x, y, seqlens, masks = batch
        x = x.to(device)
        y = y.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        loss = model(x, masks, y, training=True) 

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        if i%output_loss_freq==0: 
            logger.info(f"iteration:{epoch} of {n_epochs}, step: {i}/{len(iterator)}, NER loss: {np.mean(losses):.6f}")
            losses = []

# 验证及测试函数
def eval(model, iterator):
    model.eval()

    y_true, y_pred = [], []
    phrases_count = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, seqlens, masks = batch
            x = x.to(device)
            masks = masks.to(device)
            _, y_hat = model(x, masks, training=False) 

            for i,seqlen in enumerate(seqlens):
                phrases_count += 1
                y_true.extend(y[i,1:seqlen-1].tolist())
                y_pred.extend(y_hat[i,1:seqlen-1].tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    logger.info(f'processed {len(y_true)} tokens with {phrases_count} phrases;')

    acc,p,r,f1 = accuracy_score(y_true,y_pred),precision_score(y_true,y_pred,average='macro'), \
                    recall_score(y_true,y_pred,average='macro',zero_division=0),f1_score(y_true,y_pred,average='macro')

    logger.info(f'accuracy: {acc:.4f}, precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}')

    for idx,tag in idx2tag.items():
        if tag in [START_SYMBOL,END_SYMBOL]:
            continue
        tp = np.sum(y_pred[y_true == idx] == idx)
        fp = np.sum(y_true[y_pred == idx] != idx)
        fn = np.sum(y_pred[y_true == idx] != idx)

        _p = tp / (tp+fp)
        _r = tp / (tp+fn)
        if _p == 0 and _r ==0:
            _f1 = 0
        else:
            _f1 = 2*_p*_r/(_p+_r)

        logger.info(f'{tag}: precision: {_p:.4f}, recall: {_r:.4f}, f1: {_f1:.4f}')

    return p,r,f1

if __name__=="__main__":

    # 使用cuda但是cuda获取不到
    if use_cuda and not torch.cuda.is_available():
        raise Exception('You choose use cuda but cuda is not available.')

    os.makedirs(output_dir,exist_ok=True)   # 创建输出目录

    model = BertLstmCRF().to(device)
    model_save_path = os.path.join(output_dir, 'model.pth')


    logger.info('Initial model Done')

    train_dataset = NerDataset(train_file)
    eval_dataset = NerDataset(dev_file)
    test_dataset = NerDataset(test_file)
    logger.info('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    param_optimizer = model.named_parameters()  # 模型参数
    # 针对bert以及非bert部分设置不同的学习率
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if 'bert' in n], 'lr': bert_lr},
        {'params': [p for n, p in param_optimizer if 'bert' not in n], 'lr': lr}
        ]
    optimizer = optim.Adam(optimizer_grouped_parameters)

    logger.info('Start Train...')

    best_dev_f1 = 0
    no_improve_epoch = 0    # 验证集F1没有提示的轮数

    for epoch in range(1, n_epochs+1):  # 每个epoch对dev集进行测试

        train(model, train_iter, optimizer, epoch)

        logger.info(f"evaluate at epoch={epoch}")
        
        
        precision, recall, f1 = eval(model, eval_iter)

        if f1 > best_dev_f1:
            best_dev_f1 = f1
            logger.info(f'new best dev f1: {f1:.4f}')
            no_improve_epoch = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info('model saved')
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= early_stop:
                logger.info('Early stoping...')
                break


    logger.info('Train done, testing...')
    precision, recall, f1 = eval(model, test_iter)