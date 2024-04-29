# -*- coding: utf-8 -*-
# @Time    : 2020/7/13 上午9:51
# @Author  : PengChen
# @File    : torchtext with text classification.py


import numpy as np
import pandas as pd
import torch
from torchtext.data import Field
from torchtext.data import Iterator, BucketIterator,TabularDataset,ReversibleField
from torchtext.vocab import Vectors



import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# device = torch.device('cuda' if torch.cuda else "cpu")

## load data
root = "/home/cp/dataSet/text-classification-sample/"
train_path = "/home/cp/dataSet/text-classification-sample/train.csv"
valid_path = '/home/cp/dataSet/text-classification-sample/valid.csv'
test_path = '/home/cp/dataSet/text-classification-sample/test.csv'

data_train = pd.read_csv(train_path).head()
print(data_train)
print(data_train.columns)

# 1 定义Field对象，配置文本处理
tokenize = lambda x: x.split()
text = Field(sequential=True,lower=True,tokenize=tokenize,use_vocab=True)
label = Field(sequential= False,use_vocab=False)

# 2 定义DataSets对象，加载原始语料
tv_datafields = [("id",None),("comment_text",text),("toxic",label),('severe_toxic',label),('obscene',label),
                 ('threat',label),('insult',label),("identity_hate",label)]

train, valid = TabularDataset.splits(path= root,train='train.csv',validation='valid.csv',
                                     format = 'csv',skip_header = True,
                                     fields = tv_datafields)

test_datafields = [('id',None),('comment_text',text)]

test = TabularDataset.splits(path=test_path,format="csv",skip_header=True,fields= test_datafields)


## 3 建立词汇表
embedding_path = "/home/cp/dataSet/Embeddings/glove.6B/glove.6B.50d.txt"
vector = Vectors(name= embedding_path)
text.build_vocab(train,valid,vectors = vector,unk_init =torch.nn.init.xavier_uniform)
# text.vocab.vectors.unk_init = torch.nn.init.xavier_uniform## UNK的初始化
## 输出词汇表的相关内容
most_commom = text.vocab.freqs.most_common(10)
print(most_commom)
print(train[0])#<torchtext.data.example.Example object at 0x7fba617856d8>  是一个Example对象

print(train[0].__dict__)##对象的先关描述
print(train[0].__dict__.keys())
print(train[0].__dict__.values())
print(train[0].comment_text)## 文本信息
print(len(train[0].comment_text))#43
##标签信息
print(train[0].toxic)
print(train[0].severe_toxic)
print(train[0].obscene)
print(train[0].threat)
print(train[0].insult)
print(train[0].identity_hate)

print(text.vocab.itos[12])
print(text.vocab.stoi['it'])
print(text.vocab.stoi['pad'])

print(len(text.vocab))# 1435,嵌入词典的大小

print(text.vocab.vectors.shape)#[1435, 50]
word_vec = text.vocab.vectors[text.vocab.stoi['it']]
print(word_vec.shape)
print(word_vec)
print(text.pad_token)# <pad>
print(text.vocab.stoi[text.pad_token])# <pad> 对应的id = 1
print(text.vocab.itos[0])# <unk> 对应的id = 0

### 3 创建迭代

train_iter,valid_iter  = BucketIterator.splits((train,valid),
                           batch_size= 64,
                           device = 'cpu',
                           sort_key = lambda x: len(x.comment_text),#使用什么功能对数据进行分组。
                          sort_within_batch = False,
                           repeat = False
                           )


print(train_iter)
print(next(train_iter.__iter__()))



test_iter = Iterator.splits(datasets = test, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)

"""
这个是需要将数字化的序列转为原始的文本
TEXT = ReversibleField(sequential=True, lower=True, include_lengths=True)
for data in train_iter:
    (x, x_lengths), y = data.Text, data.Description
    orig_text = TEXT.reverse(x.data)
    print(orig_text)
"""

class BatchWrapper():
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            ## getattr 返回对象的属性
            ## getattr(object, name[, default]) name-->object
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is not None:  # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)

train_dl = BatchWrapper(train_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
valid_dl = BatchWrapper(valid_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
test_dl = BatchWrapper(test_iter, "comment_text", None)

print(train_dl)

print(next(train_dl.__iter__()))
print(len(next(train_dl.__iter__())))
print(next(train_dl.__iter__())[0].shape)

## network model
PAD_IDX = text.vocab.stoi[text.pad_token]
class Model(nn.Module):
    def __init__(self,embedding_dim = 50,hidden_dim = 120,recurrent_dropout = 0.3,num_liner =1,num_class = 6):
        super(Model,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(text.vocab),
                                      embedding_dim= embedding_dim,
                                      padding_idx= PAD_IDX )
        self.lstm = nn.LSTM(input_size= embedding_dim,hidden_size= hidden_dim,num_layers= 1,
                            dropout = recurrent_dropout)
        self.liner_layer = []
        for _ in range(num_liner- 1):
            self.liner_layer.append(nn.Linear(hidden_dim,hidden_dim))
        self.liner_layer = nn.ModuleList(self.liner_layer)
        self.out = nn.Linear(hidden_dim,num_class)


    def forward(self,x):
        #x:[sent len, batch size]
        embedding_layer = self.embedding(x)# [sent len, batch size, emb dim]
        hidden,_ = self.lstm(input= embedding_layer)# hidden = [sent len, batch size, emb dim]
        feature = hidden[-1,:,:]
        for layer in self.liner_layer:
            feature = layer(feature)
        out = self.out(feature)
        return out

model = Model()

print(model)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/text_classification')#指定路径



## optim and loss

optimizer = optim.Adam(model.parameters(),lr = 1e-3)
loss_function  = nn.BCEWithLogitsLoss()



## train model
epochs = 5
for epoch in range(epochs):
    loss_ = 0
    correct_ = 0
    # model.train() is used to put the model in "training mode",
    # which turns on dropout and batch normalization.
    model.train()#打开训练模式
    for x,y in train_dl:
        """
        对于每个批次，我们首先将梯度归零。模型中的每个参数都有一个 grad 属性，
        它存储了由准则计算出的梯度。PyTorch 不会自动删除（或 "归零"）上次梯度计算中计算出的梯度，
        因此必须手动将其归零。
        """
        optimizer.zero_grad()
        writer.add_graph(model, x)
        writer.close()
        predict = model(x)
        loss = loss_function(predict,y)#计算的损失是每个batch的损失
        loss.backward()# 计算每个参数的梯度
        optimizer.step()# 优化器算法更新参数
        loss_ += loss.item() * x.size(0)
    epoch_loss = loss_ / len(train)## len(train) 是迭代器中的批次数,,所有batch的损失/batch 数目 = 平均的batch loss
    # calculate the validation loss for this epoch
    val_loss = 0
    model.eval()#打开评估模式 this turns off dropout and batch normalization.
    for x,y in valid_dl:
        pred = model(x)
        loss = loss_function(pred,y)
        val_loss += loss.item() * x.size(0)

    val_loss /= len(valid)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))


## 测试
