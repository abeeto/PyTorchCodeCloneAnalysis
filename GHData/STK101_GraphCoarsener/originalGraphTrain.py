import GCN as GCN
import torch
import torch_geometric
import math
import numpy as np
import torch
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import argparse
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
import time
import sys

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def splits(data,num_classes,num_val,num_test,split='random'):

    if split == 'full':
        data.train_mask.fill_(True)
        data.train_mask[data.val_mask | data.test_mask] = False
        
    else:
        data.train_mask = [False]*data.num_nodes
        for c in range(num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            num_train_per_class = (int)(0.8 * len(idx))
            
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            # print("See: ",idx)
            data.train_mask = index_to_mask(idx,size = data.num_nodes)

        remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]
        # val_idx
        data.val_mask = [False]*data.num_nodes
        data.val_mask=  index_to_mask(remaining[:num_val],size = data.num_nodes)
        data.test_mask = [False]*data.num_nodes
        data.test_mask = index_to_mask(remaining[num_val:num_val + num_test],size = data.num_nodes)
    return data
def split(data, num_classes):
    indices = []
    num_test = (int)(data.num_nodes * 0.1 / num_classes)
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
        test_index = torch.cat([i[:num_test] for i in indices], dim=0)
        val_index = torch.cat([i[num_test:num_test*2] for i in indices], dim=0)
        train_index = torch.cat([i[num_test*2:] for i in indices], dim=0)
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data
#for DBLP
#splits(data,num_classes,1700,1700,split='random')

def parse_args():
    parser = argparse.ArgumentParser(description='Original Graph Training')
    parser.add_argument('--data_dir',type=str,required=False,help="Path to the dataset")
    parser.add_argument('--epochs' , type=int,required=False, default=100,help="Number of epochs to train the original graph")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    torch.cuda.empty_cache()
    # dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')
    dataset = Coauthor(root = 'data/Physics', name = 'Physics')
    data = dataset[0]
    data = split(data,5)
    # splits(data,4,1700,1700,split='random')
    
    
    # data = splits(data,15,'random') 
    model = GCN.GCN(num_features = dataset.num_features, hidden_channels=16, num_classes = dataset.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    learning_rate = 0.01
    decay = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    losses = []
    best_val_acc = 0
    best_test_acc = 0
    start = time.time()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        pred = out.argmax(1)
        criterion = torch.nn.NLLLoss()
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('In epoch {}, loss: {:.3f},train_acc:{:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
    print("Time taken to train: " , start - time.time())
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')






