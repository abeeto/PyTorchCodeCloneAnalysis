mport unicodedata
import codecs
from io import open
import itertools
import math
import sys

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def indexesFromSentence(voc, sentence):
    return [voc[word] for word in sentence.split(" ")]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value = PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for item in seq:
            if item == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l,voc):
    indexes_batch = [indexesFromSentence(voc,sentence) for sentence in l]
    lengths = torch.tensor([len(index) for index in indexes_batch])
    padlist = zeroPadding(indexes_batch)
    padvar = torch.Longtensor(padlist)
    return padvar,lengths

def outputVar(l,voc):
    indexes_batch = [indexesFromSentence(voc,sentence) for sentence in l]
    max_target_len = max([len(index) for index in indexes_batch])
    padlist = zeroPadding(indexes_batch)
    mask = binaryMatrix(padlist)
    padvar = torch.Longtensor(padlist)
    mask = torch.Longtensor(mask)
    return padvar,mask,max_target_len
 
def batch2TrainData(voc,pair_batch):
    pairs = pairs.sort(key = lambda x: len(x[0].split(" ")), reverse = True)
    inputlist = []
    outputlist = []
    for pair in pairs:
        inputlist.append(pair[0])
        outputlist.append(pair[1])
    inp, lengths = inputVar(voc,inputlist)
    output, mask, max_target_len = outputVar(voc,outputlist)
    return inp,lengths,output,mask,max_target_len
    
