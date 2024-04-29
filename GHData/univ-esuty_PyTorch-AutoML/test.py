from __future__ import print_function

from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from train import compose_net, make_dataloader

if __name__ == '__main__':
    cfg = OmegaConf.load('config/test-config.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataloader, _ = make_dataloader(cfg.data, cfg.batch_size, n_train_ratio=100)
    net = compose_net(cfg.model)
    net = net.to(device)
    net.load_state_dict(cfg.weight_path)
    
    print('Start testing...')
    net.eval()
    correct = 0
    total = 0
    gt_list = np.array([])
    pred_list = np.array([])
    
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                v_acc = 100.*correct/total
                
                gt_list = np.append(gt_list, targets.cpu().numpy())
                pred_list = np.append(pred_list, predicted.cpu().numpy())
                
                pbar.set_postfix(acc=v_acc)
                pbar.update(1)
    
    print('Making a confusion-matrix...')
    cm = confusion_matrix(gt_list.astype(np.int32), pred_list.astype(np.int32))
    cm = pd.DataFrame(data=cm, 
                      index=[chr(ord('A')+i) for i in range(26)],
                      columns=[chr(ord('A')+i) for i in range(26)])
    sns.set(rc={'figure.figsize': (32, 32)})
    sns.heatmap(cm, square=True, cbar=True,
                annot=True, cmap='Blues', robust=True, fmt='d')
    plt.yticks(rotation=0)
    plt.xlabel("PD", fontsize=13, rotation=0)
    plt.ylabel("GT", fontsize=13)
    plt.title(f"Capital alphabet classification:acc={v_acc:.2f})")
    plt.savefig(f'{cfg.weight_path}-confusion-matrix.png')
    

