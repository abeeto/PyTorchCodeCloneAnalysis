import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# embedding size of fasttext models
d_model = 300 

device = torch.device("cuda:0")
dtype = torch.float

class Language_DataSet(torch.utils.data.Dataset):

    def __init__(self, path, language_in, language_out):
        super().__init__()
        self.language_in = language_in
        self.language_out = language_out

        # load dat and vocab
        self.data = joblib.load(path+f'{language_in}_to_{language_out}.data')
        self.vocab_in = joblib.load(path+f'vocab_{language_in}.data')
        self.vocab_out = joblib.load(path+f'vocab_{language_out}.data')

        # zero padding info
        self.seq_len_in = self.vocab_in["max_sentence_len"] + 1 # additional <SOS>/<EOS> token
        self.seq_len_out = self.vocab_out["max_sentence_len"] + 1 # additional <SOS>/<EOS> token

        # precompute padding
        self.precompute_padding = torch.zeros(1, d_model).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return format: input_encoding, output_encoding, target

        x_in, x_out, target = self.data[index]
        x_in = x_in.to(device)
        x_out = x_out.to(device)
        
        # zero padding

        seq_len_in = x_in.shape[0]
        padding_length_in = self.seq_len_in - seq_len_in

        seq_len_out = x_out.shape[0]
        padding_length_out = self.seq_len_out - seq_len_out

        padding_in = self.precompute_padding.repeat([padding_length_in,1])
        padding_out = self.precompute_padding.repeat([padding_length_out,1])

        x_in = torch.cat([x_in, padding_in], dim=0)
        x_out = torch.cat([x_out, padding_out], dim=0)

        # padding target <EOS> as target for padding?
        target_len = len(target)
        padding_length = self.seq_len_out - target_len
        padding_index = target[-1] # <EOS>

        target = target + [ padding_index for i in range(padding_length)]
        target = torch.tensor(target).to(device)

        return x_in, x_out, target