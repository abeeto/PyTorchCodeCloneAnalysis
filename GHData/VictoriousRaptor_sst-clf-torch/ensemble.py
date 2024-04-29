# -*- coding: utf-8 -*-
# Ensembling 4 networks
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim 
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import Config
from TextCNN import TextCNN
from RCNN import RCNN
from RNN import myRNN, LSTMClassifier
from dataset import SSTDataset, loadGloveModel
import argparse
import time
import numpy as np
import collections
from main import collate_fn

def evaluation(data_iter, models, args):
    for model in models:
        model.eval()
    with torch.no_grad():
        # corrects = 0
        avg_loss = 0
        # total = 0
        voted_corrects = 0
        for data, label in data_iter:
            sentences = data.to(args.device, non_blocking=True)
            labels = label.to(args.device, non_blocking=True)
            batch_res = np.zeros((len(models), labels.size()[0]), dtype=np.int)
            for i, model in enumerate(models):
                logit = model(sentences)
                # torch.max(logit, 1)[1]: index
                batch_res[i] = torch.max(logit, 1)[1].view(labels.size()).cpu().data
            # vote = collections.Counter(batch_res)
            vote = [collections.Counter(batch_res[:,i]).most_common(1)[0][0] for i in range(labels.size()[0])]
            voted_corrects += (vote == label.cpu().numpy()).sum()

        size = len(data_iter.dataset)
        for model in models:
            model.train()
        return 100.0 * voted_corrects / size

def main():

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--kernel_num', type=int, default=100)
    parser.add_argument('--label_num', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--wordvec_dim', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='rcnn')
    parser.add_argument('--early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('--test-interval', type=int, default=200, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5')
    parser.add_argument('--dataset_path', type=str, default='data/dataset/')
    
    args = parser.parse_args()
    # torch.manual_seed(args.seed)[]

    start = time.time()
    wordvec = loadGloveModel('data/glove/glove.6B.'+ str(args.wordvec_dim) +'d.txt') 
    args.device = device
    args.weight = torch.tensor(wordvec.values, dtype=torch.float)
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

    # Datasets
    testing_set = SSTDataset(args.dataset_path, 'test', args.label_num, args.wordvec_dim, wordvec)
    testing_iter = DataLoader(dataset=testing_set,
                                    batch_size=args.batch_size,
                                    num_workers=0, collate_fn=collate_fn, pin_memory=True)

    print(time.time() - start)

    model_name = args.model_name.lower()

    # training_set = SSTDataset(args.dataset_path, 'train', args.label_num, args.wordvec_dim, wordvec)
    models = [TextCNN(args).to(device), LSTMClassifier(args).to(device), RCNN(args).to(device), myRNN(args).to(device)]
    models[0].load_state_dict(torch.load('model_cnn_{}_{}.ckpt'.format(args.wordvec_dim, args.label_num)))
    models[1].load_state_dict(torch.load('model_lstm_{}_{}.ckpt'.format(args.wordvec_dim, args.label_num)))
    models[2].load_state_dict(torch.load('model_rcnn_{}_{}.ckpt'.format(args.wordvec_dim, args.label_num)))
    models[3].load_state_dict(torch.load('model_rnn_{}_{}.ckpt'.format(args.wordvec_dim, args.label_num)))

    del wordvec  # Save some memory

    print(evaluation(testing_iter, models, args))
    print("Parameters:")
    delattr(args, 'weight')
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

if __name__ == "__main__":
    main()