# -*- coding: utf-8 -*-
# @Time    : 2020/7/16 上午9:27
# @Author  : PengChen
# @File    : sentiment analysis.py
# ref :https://github.com/hpanwar08/sentiment-analysis-torchtext
#      https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import re
import os

import torch
import torch.nn as nn
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field,Iterator,BucketIterator,TabularDataset
from torchtext.vocab import Vectors
import torch.nn.functional as F
import numpy as np
import random


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)##为CPU设置种子用于生成随机数
torch.cuda.manual_seed(SEED)##为当前GPU设置随机种子
torch.backen.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_root = '/home/cp/dataSet/Sentiment_Analysis'
data_path = "/home/cp/dataSet/Sentiment_Analysis/Sentiment_Analysis_Dataset.csv"

data_df = pd.read_csv(data_path,error_bad_lines = False)
data_head = data_df.head(3)
print(data_head)
print(data_df.columns)
print(set(data_df.Sentiment))## 标签只有0和1

"""
## 标签是否平衡
fig = plt.figure(figsize=(7,6))
ax = sns.barplot(x= data_df.Sentiment.unique(),y=data_df.Sentiment.value_counts())
ax.set(xlabel = "Lable")
plt.show()
# 显示类别平衡
"""

## 处理文本

data_df.drop(columns=['ItemID','SentimentSource',],inplace=True)

print(data_df.head())

def normalise_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

data_df['SentimentText'] = normalise_text(data_df['SentimentText'] )

print(data_df.head())

## 划分数据
def split_data():
    train, valid = train_test_split(data_df,test_size = 0.2,random_state = 1234,)
    return train.reset_index(drop = True), valid.reset_index(drop = True)


train_df, valid_df = split_data()

train_df.to_csv(os.path.join(data_root,'train.csv'), index=False)
valid_df.to_csv(os.path.join(data_root,'valid.csv'), index=False)

## 定义数据处理的对象
tokenizer = lambda x:x.split()

TEXT = Field(sequential=True,use_vocab=True,tokenize= tokenizer,batch_first=True,include_lengths = True)
##如何这里用了batch_first 那么后面的地方都需要注意 使用 batch_first ,,建议 在最后模型的地方进行转换
# pad_packed_sequence 和nn.LSTM nn.GRU 中都需要指定参数batch_first=True
## include_lengths = True 返回一个已经补全的最小batch的元组 (在pacK_pad_sentences使用)
##这将使batch.text现在成为一个元组，第一个元素是我们的句子（已填充的数字化张量），第二个元素是我们句子的实际长度。
LABEL  = Field(sequential=False,use_vocab=False,batch_first=True)

text_field = [('Sentiment',LABEL),("SentimentText",TEXT)]### 我们传入的字段必须和数据的列相同

train_data,valid_data = TabularDataset.splits(path= data_root,train='train.csv',
                                              validation='valid.csv',format = 'csv',
                                              fields = text_field,
                                              skip_header = True)

example = train_data[0]

print(example)#Example object
print(example.__dict__.keys())#Example对象将单个数据点的属性捆绑在一起
print(example.Sentiment)
print(example.SentimentText)##获取分析的结果 但是还没有数字化


## 加载词向量
embedding_path = "/home/cp/dataSet/Embeddings/glove.6B/glove.6B.50d.txt"
vector = Vectors(name= embedding_path)
TEXT.build_vocab(train_data,vectors = vector,min_freq = 2, unk_init =torch.nn.init.xavier_uniform)##对unk的词的初始化
# 这使torchtext遍历训练集中的所有元素，检查与TEXT字段相对应的内容，并将单词注册在其词汇表中。
# min_freq = 2 只允许出现至少2次的token 出现在我们的词汇表中,仅出现一次token转换成<UNK>,从而得到训练.
"""
为什么我们只在训练集上建立词汇表？测试任何机器学习系统时，您都不希望以任何方式查看测试集。
我们不包括验证集，这可以防止“信息泄漏”进入我们的模型，从而使我们夸大了验证/测试分数。
"""
LABEL.build_vocab(train_data)
# 词典大小
print(TEXT.vocab.vectors.shape)


## batch
train_iter, valid_iter = BucketIterator.splits(datasets = (train_data,valid_data),
                                               batch_sizes =(3,3),
                                               sort_key = lambda x: len(x.SentimentText),
                                               device = device,
                                               sort_within_batch = True,
                                               repeat = False)
"""
当sort_within_batch参数设置为True时，将根据sort_key以降序对每个小型批处理中的数据进行排序
当您要对填充序列数据使用pack_padded_sequence并将填充序列张量转换为PackedSequence对象时，sort_within_batch = True是必需的
"""

batch = next(iter(train_iter))

print(type(batch))
print(batch.Sentiment)
print(batch.SentimentText)
print(batch.SentimentText[0])## 取batch中的SentimentText的第一个句子的内容
print(batch.SentimentText[0].cpu().numpy())#转化为numpy
print(batch.dataset.fields)
print(batch.__dict__.keys())
print(batch.__dict__.values())

## 索引到句子
def idxtosent(batch, idx):
    return ' '.join([TEXT.vocab.itos[i] for i in batch.SentimentText[idx].cpu().data.numpy()])

ret0 = idxtosent(batch,0)
print(ret0)

ret1 = idxtosent(batch,1)
print(ret1)

"""
请注意，BucketIterator返回一个Batch对象而不是文本索引和标签，并且与pytorch Dataloader不同，Batch对象是不可迭代的
单个Batch对象包含一个批处理的数据，并且可以通过列名访问文本和标签
这是torchtext的一个小毛病。但这可以通过两种方式轻松克服。要么在训练循环中写一些额外的代码，
用于从Batch对象中获取数据，要么写一个围绕Batch对象的可迭代包装器，返回所需的数据。
我将采取第二种方法，因为这要干净得多。
"""


class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)#####?????和上一个相比????
            yield (X, y)

train_dl = BatchGenerator(train_iter,'SentimentText','Sentiment')###输出的妮儿是?????
valid_dl = BatchGenerator(valid_iter,'SentimentText','Sentiment')

## model
embedding_dim = len(TEXT.vocab.vectors.shape[1])
num_embeddings = len(TEXT.vocab)
n_hidden = 120
pre_trained_vec =TEXT.vocab.vectors
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding id


class SimpleGRU(nn.Module):
    def __init__(self,n_hidden,num_embeddings,embedding_dim, pretrained_vec, bidirectional = True):
        super(SimpleGRU,self).__init__()
        self.n_hidden = n_hidden
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(num_embeddings= self.num_embeddings,
                                  embedding_dim= self.embedding_dim,
                                  padding_idx = PAD_IDX)## padding 的初始化权重是0
        self.embed.weight.data.copy_(pretrained_vec)## 加载预训练的嵌入替换嵌入层的初始权重。
        #self.embedding.weight.data[PAD_IDX] = torch.zeros(self.embedding_dim) 是否需要初始化为0向量

        self.embed.weight.requires_grad = False  # make embedding non trainable

        self.gru = nn.GRU(input_size= embedding_dim,hidden_size= self.n_hidden ,
                          num_layers=1,bidirectional=self.bidirectional,
                          dropout = 0.3,
                          batch_first = True)
        self.out = nn.Linear(in_features=self.n_hidden*2,out_features=1)##二分类问题 这里是1
    def forward(self,seq,lengths):
        ## Field中设置batch_first= True 所以 seq 的shape: Batch , seq_lenth,
        bs = seq.size(0)  # batch size
        print('batch size', bs)
        self.h0 = self.init_hidden(bs)  # initialize hidden state of GRU
        print('Inititial hidden state shape', self.h.shape)

        embeds = self.embed(seq)# shape: Batch , seq_lenth,featrue_dim
        embs = pack_padded_sequence(embeds, lengths,batch_first=True)  # unpad
        """
        如果没有将初始隐藏状态作为参数传递，则默认为全零的张量。
        """
        gru_out,(hidden,cell) = self.gru(embs, self.h0)#gru返回所有时间步的隐藏状态以及最后一个时间步的隐藏状态,但是输出gru_out 是PackedSequence类型的数据

        gru_out, lengths = pad_packed_sequence(gru_out,batch_first=True)  # pad the sequence to the max length in the batch

        print('GRU output(all timesteps)', gru_out.shape)
        print(gru_out)
        print('GRU last timestep output')
        print(gru_out[-1])
        print('Last hidden state', hidden)
        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        # since it is as classification problem, we will grab the last hidden state
        outp = self.out(hidden[-1]) # hidden[-1] contains hidden state of last time step
        return F.log_softmax(outp, dim=-1)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros((2,batch_size,self.n_hidden)).to(device)
        else:
            return torch.zeros((1,batch_size,self.n_hidden)).to(device)## num_layer*num_direction, batch_size, hidden size

## model 2
class ConcatPoolingGRUAdaptive(nn.Module):

    def __init__(self,vocab_size,embedding_dim,n_hidden,pretrained_vec, bidirectional =True):
        super(ConcatPoolingGRUAdaptive, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx = PAD_IDX)
        self.embed.weight.data.copy_(pretrained_vec)# load pre-trained vector
        self.embed.weight.requires_grad = False # make embedding non trainable
        self.gru = nn.GRU(self.vocab_size,self.n_hidden,bidirectional = self.bidirectional,batch_first= True)

        self.out = nn.Linear(self.n_hidden*2, self.n_out)

    def forward(self,seq,length):
        bs = seq.size(0)
        h0 = self.init_hidden(seq.size(0))#####??????
        embed = self.embed(seq,)
        embed = pack_padded_sequence(embed,length)
        gru_out, self.h = self.gru(embed, h0)
        gru_out, lengths = pad_packed_sequence(gru_out,batch_first= True)

        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)
#         outp = self.out(torch.cat([self.h[-1],avg_pool,max_pool],dim=1))
        outp = self.out(torch.cat([avg_pool,max_pool],dim=1))
        return F.log_softmax(outp, dim=-1)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros((2,batch_size,self.n_hidden)).to(device)
        else:
            return torch.zeros((1,batch_size,self.n_hidden)).cuda().to(device)


model_SimpleGRU  = SimpleGRU(n_hidden,num_embeddings,
                             embedding_dim,
                             pre_trained_vec).to(device)


"""
创建一个函数，该函数将告诉我们模型有多少个可训练参数，以便我们可以比较不同模型之间的参数数量。
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model_SimpleGRU):,} trainable parameters')


## train model
import torch.optim as optim

def fit(train_dl,valid_dl, model, epochs = 3):
    optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-3)####???
    loss_fn = F.nll_loss
    for epoch in range(epochs):
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0
        model.train()  # 打开训练模式

        for (x,lengths),y in train_dl:## 此时的X中包含 data 和对应的长度
            lengths = lengths.cpu().numpy()
            optimizer.zero_grad()
            pred = model(x, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            pred_idx = torch.max(pred, dim=1)[1]

            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred_idx.cpu().data.numpy())
            total_loss_train += loss.item()

            train_acc = accuracy_score(y_true_train, y_pred_train)
            train_loss = total_loss_train / len(train_dl)
        if valid_dl:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
            for (x,lengths),y in valid_dl:
                pred = model(x, lengths.cpu().numpy())
                loss = loss_fn(pred, y)
                pred_idx = torch.max(pred, 1)[1]
                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(pred_idx.cpu().data.numpy())
                total_loss_val += loss.item()
            valacc = accuracy_score(y_true_val, y_pred_val)
            valloss = total_loss_val / len(valid_dl)

            print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {valloss:.4f} val_acc: {valacc:.4f}')
        else:
            print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}')

# train_model = fit(train_dl,valid_dl,model_SimpleGRU,)













