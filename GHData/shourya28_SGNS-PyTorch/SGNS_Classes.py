
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Input:
    def __init__(self,
                 input_path = 'Test.txt',
                 output_path = 'Output.txt',
                 window_size = 5,
                 neg_sample = 5):
        
        self.input_path = input_path
        self.output_path = output_path
        self.word_freq = {}
        self.word2id = {}
        self.id2word = {}
        self.sample_table = list()
        self.word_count = 0
        self.Calculate_Word_Freq()
        self.Create_Vocab()
        self.Sample_Table()
        
    def Calculate_Word_Freq(self):
        self.input_file = open(self.input_path, encoding = "utf8")
        
        for lines in self.input_file:
            lines = lines.split('|')
            for line in lines:
                line = line.split(' ')
                self.word_count += len(line)
                for word in line:
                    try:
                        self.word_freq[word] += 1
                    except:
                        self.word_freq[word] = 1

    def Create_Vocab(self):
        
        for i,w in enumerate(self.word_freq.keys()):
            self.word2id[w]=i
            self.id2word[i]=w
            
            
    def Sample_Table(self):
        freq_distribution = list()
        
        sample_table_size = 1e6
        
        for f in self.word_freq.values():
            freq_distribution.append(f)
        
        freq_distribution = (np.array(freq_distribution))**0.75
        prob_distribution = (freq_distribution)/sum(freq_distribution)
        count = np.round(prob_distribution * sample_table_size)
        
        for w_idx,i in enumerate(count):
            self.sample_table += [w_idx] * int(i)
            
        self.sample_table = np.random.shuffle(np.array(self.sample_table))
        
           
    def Gen_Neg_Sample(self, neg_sample = 5):
        return np.random.choice(self.sample_table, neg_sample)   


    def pairs(self, window_size,batch_size=None):
        if batch_size==None:
            batch_size = self.Pair_count(self,window_size)
        while pairs<batch_size:
            text = self.input_file.readlines()
            if self.input_file == '' or self.input_file == None:
                self.input_file = open(self.input_path, encoding = "utf8")
                text = self.input_file.readlines()
                
            pairs = list()
            word_idx = list()
            for lines in text:
                lines = lines.strip().split('|')
                for line in lines:
                    line = line.split(' ')
                    for word in line:
                        word_idx.append(self.word2id[word])

            for pos,u in enumerate(word_idx):
                for v in word_idx[max(i-window_size,0),min(i+window_size,len(word_idx))]:
                    if u==v: continue:
                    pairs.append((u,v))

        return pairs
                        
                        
    def Pair_Count(self,window):
        return (2*self.word_count-1)*window


class SGNS(nn.Module):
    def __init__(self, vocab_size = 100, emb_dim = 100):
        super(SGNS,self).__init__()
        self.u_emb = nn.Embedding(vocab_size, emb_dim)
        self.v_emb = nn.Embedding(vocab_size, emb_dim)
    
    def forward(self,u_idx,v_idx,v_neg_idx):
        u_emb = self.u_emb(Variable(torch.LongTensor(u_idx)))
        v_emb = self.v_emb(Variable(torch.LongTensor(v_idx)))
        
        pred = torch.mul(u_emb, v_emb).squeeze()
        pred = torch.sum(pred, dim=1)
        
        pred = F.logsigmoid(pred)
        
        neg_emb_v = self.v_emb(neg_v)
        
        neg_pred = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_pred = F.logsigmoid(-1 * neg_score)
        
        return -1 * (torch.sum(pred)+torch.sum(neg_pred))


class Word2Vec:
    def __init__(self, 
                 input_file = None, 
                 output_file = None,
                 emb_dim = 100,
                 dict_size = 10000,
                 lr = 0.01,
                 neg_sample = 5,
                 window_size = 5,
                 batch_size = None,
                 iterations = 2):
    
        self.input = Input(input_path = input_file, output_path = output_file)
        self.output = output_file
        self.emb_size = len(self.input.word2id)
        self.neg_sample = 5
        self.emb_dim = emb_dim
        self.lr = lr
        self.window_size = window_size
        pairs = self.input.Pair_Count(window_size)
        self.batch_size = batch_size
        if batch_size != None:
            self.batch_size = batch_size
            self.batch_count = pairs/self.batch_size
        else:
            self.batch_count = 1
        self.model = SGNS(self.emb_size,self.emb_dim)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr)
        
        
    def trainLoop(self):
        for i in range(self.batch_count):
            pairs = self.input.pairs(self.batch_size, self.window_size)
            
            u_idx = Variable(torch.Tensor([pair[0] for pairs in pair]).long())
            v_idx = Variable(torch.Tensor([pair[1] for pairs in pair]).long())
            v_neg = Variable(torch.Tensor(self.input.Gen_Neg_Sample(self.neg_sample)).long())
            
            if torch.cuda.is_available():
                u_idx = u_idx.cuda()
                v_idx = v_idx.cuda()
                v_neg = v_neg.cuda()
            
            loss = self.model.forward(u_idx, v_idx, v_neg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        out = open("EngEmbed.txt", mode = 'a', encoding = "utf8")
        for key,value in input.word2id.items():
            out.write(key," ",model.u_emb(value),"\n")
        
        
if __name__ == '__main__':
    word2vec = Word2Vec(input_file = "EngClean.txt", output_file = "EngEmbed.txt")
    word2vec.trainLoop()

