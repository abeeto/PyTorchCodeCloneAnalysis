import os
import argparse
import torch
import datetime

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BinarizingAutoencoder
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--embedding', type=str, default='./data/glove.6B.50d.txt', help="embedding path")
parser.add_argument('--unk', type=str, default='<unk>', help="unknown token")
parser.add_argument('--batch_size', type=int, default=256, help="batch size of the dataset")
parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
parser.add_argument('--hidden_dim', type=int, default=100, help="dimension of hidden layers")
parser.add_argument('--lambda_reg', type=float, default=0.5, help="regularization parameter of Frobenius term")
parser.add_argument('--no_cuda', action='store_true', default=False, help='disable the gpu')
parser.add_argument('--save_dir', type=str, default='snapshot', help='where to save the snapshot')
args = parser.parse_args()

class dataset():
    def __init__(self, args):
        self.data = []
        self.stoi = {}
        self.embedding = args.embedding
        self.unk = args.unk

    def prepare(self):
        with open(args.embedding, "rb") as f:
            f.readline()
            idx = 0
            for line in f:
                dataline = line.rstrip().split()
                try:
                    key = dataline[0].decode("utf-8")
                    vector = list(map(float, dataline[1:]))
                    if (len(key) > 1) & (key != self.unk):
                        self.data.append(vector)
                        self.stoi.update({key: idx})
                        idx += 1
                except IndexError:
                    pass

        self.itos = {self.stoi[word]: word for word in self.stoi.keys()}

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class to train binarized auto-encoder

    Arguments:
        word index (LongTensor): word indices
        word embeddings (FloatTensor): corresponding word embeddings
    """

    def __init__(self, word_index, word_embedding):
        assert word_index.size(0) == word_embedding.size(0)
        self.word_index = word_index
        self.embedding = word_embedding

    def __getitem__(self, item):
        return self.word[item], self.embedding[item]

    def __len__(self):
        return self.word.size()[0]

print("\nLoading data...")

dataset = dataset(args)
dataset.prepare()

args.vocab_size = len(dataset.data) + 1
args.embed_dim = len(dataset.data[0])

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))