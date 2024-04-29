import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time

import data as d
from model import LSTMTagger

parser = argparse.ArgumentParser(description='Chinese Word Segmentation Model')
parser.add_argument('--data', type=str, default='./data/',
                    help='location of the data corpus')
# parser.add_argument('--model', type=str, default='LSTM',
#                     help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=64,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
# parser.add_argument('--lr', type=float, default=20,
#                     help='initial learning rate')
# parser.add_argument('--clip', type=float, default=0.25,
#                     help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=64,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
# parser.add_argument('--tied', action='store_true',
#                     help='tie the word embedding and softmax weights')
parser.add_argument('--bidirect', action='store_true',
                    help='use bi-directional')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use CUDA')
parser.add_argument('--log_interval', type=int, default=150,
                    help='report interval')
# parser.add_argument('--save', type=str, default='model.pt',
#                     help='path to save the final model')
# parser.add_argument('--onnx-export', type=str, default='',
#                     help='path to export the final model in onnx format')
args = parser.parse_args()

TAG_CLASS = 3

if torch.cuda.is_available() and not args.no_cuda:
    device = torch.device('cuda')
    print("using CUDA")
else:
    device = torch.device('cpu')
    print('using CPU')

corpus = d.Corpus(args.data, device, args.batch_size, args.seq_len)
dict = corpus.dictionary
num_of_train_batches = corpus.total_num_of_train_batches


def accuracy(pred, target):
    mask = target != 2
    total_num = mask.sum()
    p, i = pred.max(2)
    num_correct = (target[mask] == i[mask]).sum()
    return num_correct.item(), total_num.item()

#### train ####
model = LSTMTagger(args.emsize, args.nhid, args.batch_size, len(dict), TAG_CLASS, args.nlayers, args.bidirect, args.dropout).to(device)
loss_fn = nn.CrossEntropyLoss(size_average=False).to(device)
optimizer = optim.Adam(model.parameters())

def evaluate():
    model.eval()
    with torch.no_grad():
        data = corpus.test_data_batched
        labels = corpus.test_label_batched
        pred = model(data)
        correct, total = accuracy(pred, labels)
        print('accuracy = {:.4f}'.format(correct / total))

def train(batch):
    model.train()
    model.zero_grad()
    model.hidden = model.init_hidden()
    data, labels = corpus.get_train_batch(batch)
    pred = model(data)

    loss = loss_fn(pred.view(-1, TAG_CLASS), labels.view(-1))
    correct, total = accuracy(pred, labels)

    loss.backward()
    optimizer.step()

    return loss, correct / total

# train loop
for epoch in range(args.epochs):
    for batch in range(num_of_train_batches):
        loss, acc = train(batch)
        if batch % args.log_interval == 0:
            print('epoch {} of {}, batch {} of {} | loss = {:.4f} | accuracy = {:.4f}'.format(epoch + 1,\
            args.epochs, batch, num_of_train_batches, loss, acc))
            print('eval:', end='')
            evaluate()
            print()

print('after training')
print('---- FINAL EVALUATE ----')
evaluate()
