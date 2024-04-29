#!usr/bin/python

import torch
from torchtext import data

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudann.deterministic= True

#第一步：导入数据集
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

from torchtext import datasets
train_data ,test_data = datasets.IMDB.splits(TEXT,LABEL)
#print(vars(train_data.examples[0]))  #查看数据集

#第二步：训练集划分为训练集和验证集
import random
train_data,valid_data = train_data.split(random_state=random.seed(SEED))  #默认split_ratio = 0.7

#第三步：用训练集建立vocabualry，就是把每个单词一一映射到一个数字
TEXT.build_vocab(train_data,max_size=25000,vectors='glove.6B.100d',unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

#第四步：创建iterators，每个iteration都会返回一个batch的样本
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_availabel else 'cpu')
train_iterator,valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size=BATCH_SIZE,
    device = device
)

#创建模型
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,
                 n_layers,bidirectional,dropout,pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,
                           bidirectional=bidirectional,dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,text):
        embedded = self.droput(self.embedding(text))
        output,(hidden,cell) = self.rnn(embedded)
        hidden = self.droput(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
        return self.fc(hidden.squeeze(0))

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model = RNN(INPUT_DIM,EMBEDDING_DIM,HIDDEN_DIM,OUTPUT_DIM,
            N_LAYERS,BIDIRECTIONAL,DROPOUT,PAD_IDX)

#初始化参数
pretrained_embeddings = TEXT.vocab.vectors #25002*100
model.embedding.weigth.data.copy_(pretrained_embeddings)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros[EMBEDDING_DIM]

#训练
import torch.optim as optim
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters())
model = model.to(device)

def binary_accuracy(preds,y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model,iterator,optimizer,criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions,batch.label)
        acc = binary_accuracy(predictions,batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)

    return epoch_loss / total_len, epoch_acc / total_len

def evaluate(model,iterator,criterion):
    epoch_loss =0
    epoch_acc = 0
    total_len = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions,batch.label)
            acc = binary_accuracy(predictions,batch.label)
            epoch_loss += loss.item() * len(batch.label)
            epoch_acc += acc.item() * len(batch.label)
            total_len += len(batch.label)
    model.train()
    return epoch_loss / total_len, epoch_acc / total_len

N_EPOCHS = 10
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss,train_acc = train(model,train_iterator,optimizer,criterion)
    valid_loss,valid_acc = evaluate(model,valid_iterator,criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'textC-lstm.pt')
    print('Epoch:{}'.format(epoch))
    print('Train loss:{},Train acc:{}'.format(train_loss,train_acc))
    print('Val loss:{},Val acc:{}'.format(valid_loss,valid_acc))
    print(' ')

#载入模型，在测试集上进行评估
model.load_state_dict(torch.load('wordavgmodel.pt'))
test_loss,test_acc = evaluate(model,test_iterator,criterion)
print("test_loss:{},test_acc:{}".format(test_loss,test_acc))
