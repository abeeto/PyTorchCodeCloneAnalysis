import argparse
import pandas as pd
import numpy as np

import torch
from  torchfm.model.afn import AdaptiveFactorizationNetwork
from torch.utils.data import DataLoader
from dataloader import TrainDataset
from sklearn.metrics import mean_absolute_error

from test import unscale


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', type=str, default='./out.csv')
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--embed_dims', type=int, default=16)
    parser.add_argument('--LNN_dim', type=int, default=1500)
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()
    return args


def train_one_epoch(model, loader, criterion, device, optimizer):

    model.train()
    model.to(device)
    
    for i, (fields, targets) in enumerate(loader):

        fields, targets = fields.to(device), targets.to(device)
        y = model(fields)
        
        loss = criterion(y, targets.float())
        if i% 100 == 0:

            print (loss)
            
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            torch.save(model.state_dict(), './model_inside.pth')

    
def validate_model(model, loader, device):

    model.to(device)
    model.eval()

    y_list, x_list = list(), list()
    un_ylist, un_xlist = list(), list()
    
    for i, (fields, targets) in enumerate(loader):

        fields, targets = fields.to(device), targets.to(device)
        y = model(fields)
        
        x_list.extend(y.tolist())
        y_list.extend(targets.tolist())

        scaled_x = [unscale(element) for element in y.tolist()]
        scaled_y = [unscale(element) for element in targets.tolist()]

        un_xlist.extend(scaled_x)
        un_ylist.extend(scaled_y)
        
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    print ('mean absolute error')
    print (mean_absolute_error(y_list, x_list))
    print (mean_absolute_error(un_ylist, un_xlist))

        
if __name__ == '__main__':

    args = get_args()

    field_dims = [49, 50, 50, 7, 8, 20, 20, 20, 20, 20, 20, 20, 50, 50, 20, 20, 10, 10, 5, 3, 3, 5]
    model = AdaptiveFactorizationNetwork(field_dims=field_dims, embed_dim=args.embed_dims, LNN_dim=args.LNN_dim,
                                         mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))

    #criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr,
                                 weight_decay=1e-6)

    all_data = pd.read_csv(args.input_csv)

    train_data = all_data[all_data.flag < 8]
    val_data = all_data[all_data.flag >= 8]

    train_set = TrainDataset(train_data)
    val_set = TrainDataset(val_data)

    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _ in range(args.epochs):
        train_one_epoch(model, train_loader, criterion, device, optimizer)
        validate_model(model, val_loader, device)

        torch.save(model.state_dict(), 'model.pth')
