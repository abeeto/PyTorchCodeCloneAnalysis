from collections import Counter, defaultdict
import os
import matplotlib.pyplot as pltb
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pickle
from struct import unpack
import argparse
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch.multiprocessing as mp
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

class GloveDatasetIter(IterableDataset):
    
    def __init__(self, coocurr_file, start=0, end=None):
        self.coocurr_file = open(coocurr_file, 'rb')
        self.coocurr_file.seek(0, 2)
        file_size = self.coocurr_file.tell()
        if file_size % 16 != 0:
            raise Exception('Unproper file. The file need to be the output coocurrence.bin or coocurrence.shuf.bin of offical glove.')
        if end is None:
            self.end = file_size // 16
        else:
            self.end = end
        self.start = start
        print(f'Iterable dataset loader created. start point:{self.start}, end point:{self.end}, size:{self.end - self.start}')
    
    def __len__(self):
        return self.end - self.start
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # single-process data loading, reaturn the full iterator
            iter_start = self.start
            iter_end = self.end
        else: # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for p in range(iter_start, iter_end):
            self.coocurr_file.seek(p*16)
            chunk = self.coocurr_file.read(16)
            w1, w2, v = unpack('iid', chunk)
            yield v, w1, w2

class GloveDataset(Dataset):
 
    def __init__(self, coocurr_file, start=0, end=None):
        coocurr_file = open(coocurr_file, 'rb')
        coocurr_file.seek(0, 2)
        file_size = coocurr_file.tell()
        if file_size % 16 != 0:
            raise Exception('Unproper file. The file need to be the output coocurrence.bin or coocurrence.shuf.bin of offical glove.')
        if end is None:
            end = file_size // 16
        self.len = end - start
        print(f'dataset loader created. start point:{start}, end point:{end}, size:{self.len}')
        self.v_array = np.zeros(self.len, dtype=np.float32)
        self.w1_array = np.zeros(self.len, dtype=np.int32)
        self.w2_array = np.zeros(self.len, dtype=np.int32)
        coocurr_file.seek(start * 16)
        for i in range(self.len):
            chunk = coocurr_file.read(16)
            w1, w2, v = unpack('iid', chunk)            
            self.v_array[i] = v
            self.w1_array[i] = w1
            self.w2_array[i] = w2
        coocurr_file.close()

    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        return self.v_array[i], self.w1_array[i], self.w2_array[i]

class glove(Function):
    @staticmethod
    def forward(ctx, w_i, w_j, b_i, b_j, x_ij, x_max, alpha):
        w_i, w_j, b_i, b_j = w_i.detach(), w_j.detach(), b_i.detach(), b_j.detach()
        diff = torch.sum(w_i * w_j, dim=1)
        diff += b_i + b_j - torch.log(x_ij)
        # nan and inf change to 0, according to line #131-#135 in the original c implementation
        diff[torch.isnan(diff)] = 0
        diff[torch.isinf(diff)] = 0
        fdiff = torch.where(x_ij > x_max, diff, torch.pow(x_ij / x_max, alpha) * diff)
        batch_cost = fdiff * diff * 0.5
        ctx.save_for_backward(w_i, w_j, b_i, b_j, fdiff)
        return torch.mean(batch_cost)
    @staticmethod
    def backward(ctx, grad_output):
        w_i, w_j, b_i, b_j, fdiff = ctx.saved_tensors
        grad_b_i = grad_b_j = fdiff
        fdiff = fdiff.unsqueeze(dim=1).repeat(1, w_i.size()[1])
        grad_w_i = fdiff * w_j
        grad_w_j = fdiff * w_i
        return grad_w_i, grad_w_j, grad_b_i, grad_b_j, None, None, None


def get_init_emb_weight(vocab_size, emb_size):
    init_width = 0.5 / emb_size
    init_weight = np.random.uniform(low=-init_width, high=init_width, size=(vocab_size, emb_size))
    return init_weight


class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, x_max, alpha):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wi.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, embedding_dim)))
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.wj.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, embedding_dim)))
        self.bi = nn.Embedding(vocab_size, 1)
        self.bi.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, 1)))
        self.bj = nn.Embedding(vocab_size, 1)
        self.bj.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, 1)))        
        self.x_max = x_max
        self.alpha = alpha
        
    def forward(self, i_indices, j_indices, x_ij):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        
        return glove.apply(w_i, w_j, b_i, b_j, x_ij, self.x_max, self.alpha)


def train(rank, args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    if args.threads == 0:
         start = 0
         end = None
    else:
         start = (args.num_lines // args.threads) * rank
         end = start + args.lines_per_thread[rank]
    glove_dataset = GloveDataset(args.coocurr_file, start=start, end=end)
    if isinstance(glove_dataset, IterableDataset):
        train_loader = DataLoader(glove_dataset, batch_size=args.batch_size, **dataloader_kwargs)
    else:
        train_loader = DataLoader(glove_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, **dataloader_kwargs)
    optimizer = optim.Adagrad(model.parameters(), lr=args.eta, initial_accumulator_value=1.0)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (x_ij, i_idx, j_idx) in enumerate(data_loader):
        x_ij = x_ij.float().to(device)
        i_idx = i_idx.long().to(device)
        j_idx = j_idx.long().to(device)
        optimizer.zero_grad()
        output = model(i_idx, j_idx, x_ij)
        output.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{} {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                datetime.now(), pid, epoch, batch_idx * len(x_ij), len(data_loader.dataset),
                100. * (batch_idx * len(x_ij))/ len(data_loader.dataset), output.item()))


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    
    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    vocab_file = open(args.vocab_file, 'r')
    vocab_size = sum(1 for line in vocab_file)
    vocab_file.close()
    print(f'read {vocab_size} words.')
    EMBED_DIM = args.emb_size
    model = GloveModel(vocab_size + 1, EMBED_DIM, args.x_max, args.alpha).to(device)
    model.share_memory()
    
    if args.threads == 0:
        train(0, args, model, device, dataloader_kwargs)
    else:
        glove_dataset = GloveDatasetIter(args.coocurr_file)
        num_lines = len(glove_dataset)
        del glove_dataset
        num_threads = args.threads
        lines_per_thread = []
        for a in range(num_threads-1):
            lines_per_thread.append(num_lines // num_threads)
        lines_per_thread.append(num_lines // num_threads + num_lines % num_threads)
        args.num_lines = num_lines
        args.lines_per_thread = lines_per_thread
    
        processes = []
    
        for rank in range(args.threads):
            p = mp.Process(target=train, args=(rank, args, model, device, dataloader_kwargs))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    torch.save(model.state_dict(), args.model_name)
    print(f'Model saved in: {args.model_name}.')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab_file', type=str, help='path to the vocab file', required=True)
    parser.add_argument('-c', '--coocurr_file', type=str, help='path to the coocurr file you can get it by offical glove', required=True)
    parser.add_argument('--emb_size', type=int, help='embedding size', default=200)
    # batch_size, eta tuned for this implementation and jpwiki
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.75)
    # xmax same to the offical glove example (not their default 100),
    # this leads to the similar performance in the sim_eval task with gensim skip-gram
    parser.add_argument('--x_max', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('-m', '--model_name', type=str, help='saving model path', required=True)
    parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    args = parser.parse_args()
    main(args)
