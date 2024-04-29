import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import pickle as pkl
from torch.utils.data import Dataset

class SkipGramDataset(Dataset):

  def __init__(self, datafile):
    data = pkl.load(open(datafile, 'rb'))
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

class EmbeddingNN(nn.Module):
  """ single hidden layer embedding model"""
  def __init__(self, voc_size, emb_size=300, init_with=None):
    super(EmbeddingNN, self).__init__()
    padding_idx = 0
    self.voc_size = voc_size
    self.emb_size = emb_size
    self.iembeddings = nn.Embedding(self.voc_size, self.emb_size)
    self.oembeddings = nn.Embedding(self.voc_size, self.emb_size)
    # pylint: disable=no-member
    if init_with is not None:
      assert init_with.shape == (voc_size, emb_size)
      self.iembeddings.weight = nn.Parameter(FloatTensor(init_with))
    else:
      self.iembeddings.weight = nn.Parameter(FloatTensor(voc_size, emb_size).uniform_(-1, 1))
    self.oembeddings.weight = nn.Parameter(FloatTensor(voc_size, emb_size).uniform_(-1, 1))
    # pylint: enable=no-member
    self.iembeddings.weight.requires_grad = True
    self.oembeddings.weight.requires_grad = True


  def forward(self, data):
    """"""
    return self.forward_i(data)


  def forward_i(self, data):
    """ get input vectors"""
    idxs = Variable(LongTensor(data))
    idxs = idxs.cuda() if self.iembeddings.weight.is_cuda else idxs
    return self.iembeddings(idxs)


  def forward_o(self, data):
    """ get output vectors"""
    idxs = Variable(LongTensor(data))
    idxs = idxs.cuda() if self.oembeddings.weight.is_cuda else idxs
    return self.oembeddings(idxs)


  def get_emb_dim(self):
    return self.emb_size


class SkipGram(nn.Module):
  """"""

  def __init__(self, emb_nn, n_negs=64, weights=None):
    super(SkipGram, self).__init__()
    self.emb_model = emb_nn
    self.voc_size = emb_nn.get_emb_dim()
    self.n_negs = n_negs
    self.neg_sample_weights = None
    if weights is not None:
      wf = np.power(weights, 0.75) # pylint: disable=no-member
      wf = wf / wf.sum()
      self.neg_sample_weights = FloatTensor(wf)


  def forward(self, data):
    """ data is a list of pairs"""
    batch_size = len(data[0])
    iwords = data[0]
    owords = data[1]
    if self.neg_sample_weights is not None:
      # pylint: disable=no-member
      nwords = t.multinomial(self.neg_sample_weights,
                             batch_size * self.n_negs,
                             replacement=True).view(batch_size, -1)
    else:
      nwords = FloatTensor(batch_size, self.n_negs).uniform_(0, self.voc_size - 1).long()

    ivectors = self.emb_model.forward_i(iwords).unsqueeze(2)
    ovectors = self.emb_model.forward_o(owords).unsqueeze(1)
    nvectors = self.emb_model.forward_o(nwords).neg() # important

    # pylint: disable=no-member
    oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log()
    nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, 1, self.n_negs).sum(2).mean(1)
    return -(oloss + nloss).mean()
