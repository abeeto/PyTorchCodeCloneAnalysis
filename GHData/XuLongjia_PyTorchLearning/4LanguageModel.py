#!usr/bin/python
#语言模型

import torchtext
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000

TEXT = torchtext.data.Field(lower=True)
train,val,test = torchtext.datasets.LanguageModelingDataset.splits(path='.',
    train='text8.train.txt',validation='text8.dev.txt',test='text8.test.txt',text_field=TEXT)
TEXT.build_vocab(train,max_size=MAX_VOCAB_SIZE)
print("vocabulary size : {}".format(len(TEXT.vocab)))

VOCAB_SIZE = len(TEXT.vocab)
train_iter,val_iter,test_iter = torchtext.data.BPTTIterator.splits(
    (train,val,test),batch_size=BATCH_SIZE,device=-1,bptt_len=32,repeat=False,shuffle=True
)
#模型的输入是一串文字，模型的输出也是一串文字，他们之间线差一个位置，因为语言模型的目标是根据之前的单词预测下一个单词

# it = iter(train_iter)
# batch = next(it)
# print(' '.join([TEXT.vocab.itos[i] for i in batch.text[:,1].data]))
# print(' '.join(TEXT.vocab.itos[i] for i in batch.target[:,1].data))

#定义模型
class RNNModel(nn.Module):
    '''一个简单的神经网络'''

    def __init__(self,rnn_type,ntoken,ninp,nhid,nlayers,dropout=0.5):
        '''
        该模型包含以下几层：
        词嵌入层
        一个循环神经网络层
        一个线性层，从hidden state到输出单词表
        一个dropout层，用来做regularization
        '''
        super(RNNModel,self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken,ninp)  #embedding的数量和维度
        if rnn_type in ['LSTM','GRU']:
            self.rnn = getattr(nn,rnn_type)(ninp,nhid,nlayers,dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH':'tanh','RNN_RELU':'relu'}[rnn_type]
            except:
                raise ValueError(""" An invalid option for `--model` was supplied,
                options are ['LSTM','GRU','RNN_TANH','RNN_RELU']
                """)
            self.rnn = nn.RNN(ninp,nhid,nlayers,nonlinearity=nonlinearity,dropout=dropout)
        self.decoder = nn.Linear(nhid,ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self,input,hidden):
        '''
        forward pass:
        -word embedding
        -输入循环神经网络
        -一个线性层 从hidden state转换成输出单词表
        '''
        emb = self.drop(self.encoder(input))
        output,hidden = self.rnn(emb,hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1),output.size(2)))
        return decoded.view(output.size(0),output.size(1),decoded.size(1)),hidden

    def init_hidden(self,bsz,requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (weight.new_zeros((self.nlayers,bsz,self.nhid),requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers,bsz,self.nhid),requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers,bsz,self.nhid),requires_grad=requires_grad)

model = RNNModel("LSTM",VOCAB_SIZE,EMBEDDING_SIZE,EMBEDDING_SIZE,2,dropout=0.5)
if USE_CUDA:
    model = model.cuda()

def evaluate(model,data):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE,requires_grad=False)
        for i,batch in enumerate(it):
            data,target = batch.text,batch.target
            if USE_CUDA:
                data,target = data.cuda(),target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output,hidden = model(data,hidden)
            loss = loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item()*np.multiply(*data.size())
    loss = total_loss/total_count
    model.train()
    return loss

#Remove this part
def repackage_hidden(h):
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.5)

import copy
GRAD_CLIP = 1.
NUM_EPOCHS = 2
val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i,batch in enumerate(it):
        data,target = batch.text,batch.target
        if USE_CUDA:
            data,target = data.cuda(),target.cuda()
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output,hidden = model(data,hidden)
        loss = loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        optimizer.step()
        if i%1000 ==0:
            print("epoch",epoch,"iter",i,"loss",loss.item())

        if i %10000 ==0:
            val_loss = evaluate(model,val_iter)

            if len(val_losses) ==0 or val_loss <min(val_losses):
                print("best model,val loss:",val_loss)
                torch.save(model.state_dict(),"lm-best.th")
            else:
                scheduler.step()
                optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
            val_losses.append(val_loss)

#训练完成以后如何载入模型？
best_model = RNNModel("LSTM",VOCAB_SIZE,EMBEDDING_SIZE,EMBEDDING_SIZE,2,dropout=0.5)
if USE_CUDA:
    best_model = best_model.cuda()
best_model.load_state_dict(torch.load("lm-best.th"))

#使用最好的模型在valid数据上计算perplexity
val_loss = evaluate(best_model,val_iter)
print('perplexity:',np.exp(val_loss))

#使用最好的模型在测试数据上计算perplexity
test_loss = evaluate(best_model,test_iter)
print('perplexity:',np.exp(test_loss))

#使用训练好的模型生成一些句子
hidden = best_model.init_hidden(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE,(1,1),dtype = torch.long).to(device)
words = []
for i in range(100):
    output,hidden = best_model(input,hidden)
    word_weights = output.squeeze().exp().cpu()
    word_idx = torch.multinomial(word_weights,1)[0]
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(' '.join(words))
