# * coding:utf-8 *
#@author    :mashagua
#@time      :2019/5/2 11:04
#@File      :lesson_3.py
#@Software  :PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import pandas as pd
import math
import scipy
USE_CUDA=torch.cuda.is_available()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)
C=3
K=100
NUM_EPOCHS=2
MAX_VOCAB_SIZE=30000
BATCH_SIZE=128
LEARNING_RATE=0.003
EMBEDDING_SIZE=100

def word_tokenize(text):
    return text.split()

with open("data/text8/text8") as fin:
    text=fin.read()
text=text.split()
#unk的个数
vocab=dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab["<unk>"]=len(text)-np.sum(list(vocab.values()))
idx_to_word=[word for word in vocab.keys()]
word_to_idx={word:i for word,i in enumerate(idx_to_word)}
word_counts=np.array([count for count in vocab.values()],dtype=np.float32)
word_fre=word_counts/np.sum(word_counts)
word_fre=word_fre**(3./4.)
word_fre=word_fre/np.sum(word_fre)
VOCAB_SIZE=len(vocab.keys())

