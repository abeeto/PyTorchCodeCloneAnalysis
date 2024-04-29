import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from nltk import word_tokenize
from gensim.models import Word2Vec

from preprocess import *
from plots import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LEN = 200
BATCH_SIZE = 256

def prep_word_vectors():
    word2vec = Word2Vec.load('emb/w2vwiki2018.model')
    word_vectors = np.concatenate((np.zeros((3, word2vec.wv.syn0.shape[1])), word2vec.wv.syn0), axis=0)
    word2idx = {'SOS': 0, 'EOS': 1, 'UNK': 2}
    idx2word = ['SOS', 'EOS', 'UNK']
    idx2word.extend(word2vec.wv.index2word)
    for w in word2vec.wv.vocab:
        word2idx[w] = word2vec.wv.vocab[w].index + 3

    return word2idx, idx2word, word_vectors

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, word_vectors, drop_rate=0.2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(word_vectors.shape[1], hidden_size, bidirectional=True)
        self.out_1 = nn.Linear(2*hidden_size, hidden_size)
        self.drop_1 = nn.Dropout(drop_rate)
        self.out_2 = nn.Linear(hidden_size, output_size)

        #embedding layer
        self.embedding = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(word_vectors))
        self.embedding.weight.requires_grad = False

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden) 
        #GRU input dim is (seq_len, batch, input_size)
        #output is (seq_len, batch, num_directions*hidden_size)
        #hidden is (num_layers*num_directions, batch, hidden_size)
        out = self.out_1(output.squeeze(1))
        out = F.relu(out)
        out = self.drop_1(out)
        out = self.out_2(out)
        out = torch.sigmoid(out)

        return output, hidden, out

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)

    def initOut(self):
        return torch.zeros(1, self.output_size, device=device)

class NLU_classifier():
    def __init__(self):
        #load word vectors
        self.word2idx, self.idx2word, word_vectors = prep_word_vectors()
        self.vocab_size = len(self.idx2word)
        self.label2idx, self.idx2label, _, self.dataset = clean_data()
        self.label_size = len(self.idx2label)
        self.encoder = EncoderRNN(self.vocab_size, 64, self.label_size, word_vectors).to(device)

    def sent2tensor(self, sent_tok):
        word_list = [self.word2idx[w] if w in self.word2idx else UNK_token for w in sent_tok]
        return torch.tensor(word_list, device=device).long()

    def label2list(self, labels):
        return torch.tensor([self.label2idx[l] for l in labels], device=device).long()
    
    def label2vec(self, label_list):
        vec = torch.zeros(self.label_size, device=device)
        vec[label_list] = 1.0
        return vec

    def loss_func(self, target, output):
        return -torch.mean(torch.sum(target*torch.log(output) + (1-target)*torch.log(1-output), dim=1))

    def train_test_split(self, val_r=0.05, test_r=0.05, rand=True):
        total_len = len(self.dataset)
        rand_seq = np.arange(total_len)
        if rand == True:
            np.random.shuffle(rand_seq)
        train_size = math.floor((1-val_r-test_r)*total_len)
        val_size = math.floor(val_r*total_len)
        train_seq = rand_seq[0:train_size]
        val_seq = rand_seq[train_size:train_size+val_size]
        test_seq = rand_seq[train_size+val_size:]
        return train_seq, val_seq, test_seq

    def train_step(self, input_tensor, target_tensor, max_length=MAX_LEN):
        encoder_hidden = self.encoder.initHidden() #zero for 1st token
        predict_tensor = self.encoder.initOut()
        input_length = input_tensor.size(0)
        target_tensor = self.label2vec(target_tensor).unsqueeze(0)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden, predict_tensor = self.encoder(input_tensor[ei], encoder_hidden)

        loss = self.loss_func(target_tensor, predict_tensor)
        return loss

    def eval(self, input_tensor):
        encoder_hidden = self.encoder.initHidden() #zero for 1st token
        predict_tensor = self.encoder.initOut()
        input_length = input_tensor.size(0)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden, predict_tensor = self.encoder(input_tensor[ei], encoder_hidden)

        return predict_tensor

    def load_model(self, model_path="nlu_encoder.pt"):
        self.encoder.load_state_dict(torch.load(model_path))
        
    def train(self, lr=1e-4, epochs=100):
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=lr)

        data_x = [self.sent2tensor(p[0]) for p in self.dataset]
        data_y = [self.label2list(p[1]) for p in self.dataset]
        #print (len(data_x), len(data_y))

        
        train_seq, val_seq, test_seq = self.train_test_split()

        #torch_dataset = Data.TensorDataset(data_x, data_y)
        #train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=1, shuffle=True, num_workers=2)
        for e in range(epochs):
            epoch_loss = 0
            #train
            start = time.time()
            for step in range(math.floor(len(train_seq)/BATCH_SIZE)):
                next_batch = train_seq[step*BATCH_SIZE:(step + 1)*BATCH_SIZE]
                loss = 0
                for s in next_batch:
                    loss += self.train_step(data_x[s], data_y[s])
                
                self.encoder_opt.zero_grad()
                loss.backward()
                self.encoder_opt.step()
                epoch_loss += loss.item()
                loss = 0
                #print (step)

            print('%s (%d epochs %d%% done) loss = %.4f' % (timeSince(start, e+1/epochs), e+1, e+1/epochs* 100, epoch_loss))
            torch.save(self.encoder.state_dict(), 'nlu_encoder.pt')
            
            #validate
            correct_predictions = 0
            total_predictions = 0
            for s in range(math.floor(len(val_seq))):
                predict_tensor = self.eval(data_x[s])
                result = torch.topk(predict_tensor, 3, dim=1)[1].squeeze(0)
                for l in result:
                    if l in data_y[s]:
                        correct_predictions += 1
                total_predictions += 3

            print ('val_prec@3 = %.4f %%' % (correct_predictions/total_predictions*100))



    # def eval(self, input_tensor):
    #     encoder_hidden = self.encoder.initHidden() #zero for 1st token
    #     predict_tensor = self.encoder.initOut()
    #     input_length = input_tensor.size(0)
        
    #     loss = 0
    #     for ei in range(input_length):
    #         encoder_output, encoder_hidden, predict_tensor = self.encoder(input_tensor[ei], encoder_hidden)

    #     val, idx = predict_tensor.max(0)
    #     return idx

clf = NLU_classifier()
clf.train()