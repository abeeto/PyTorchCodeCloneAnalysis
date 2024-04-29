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

#第五步：创建word averaging模型
import torch.nn as nn
import torch.nn.functional as F

class WordAVGModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,output_dim,pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim,output_dim)

    def forward(self, text):
        # text:[sent_length,batch size]
        emebedded = self.embedding(text)     #embedded : [sent_length,batch size ,embedded dim]
        emebedded = emebedded.permute(1,0,2)  #[batch size ,sent_length,embedded dim]
        pooled = F.avg_pool2d(emebedded,(emebedded.shape[1],1)).squeeze(1)
        return self.fc(pooled)
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = WordAVGModel(INPUT_DIM,EMBEDDING_DIM,OUTPUT_DIM,PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("the model has {} trainable parameters.".format(count_parameters(model)))

#第六步：初始化参数
pretrained_embeddings = TEXT.vocab.vectors #25002*100
#print(pretrained_embeddings)
#print(pretrained_embeddings.size())
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros[EMBEDDING_DIM]

#第七步 定义训练模型需要的函数
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

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



# 第八步：开始训练模型
N_EPOCHS = 10
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss,train_acc = train(model,train_iterator,optimizer,criterion)
    valid_loss,valid_acc = evaluate(model,valid_iterator,criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),'wordavgmodel.pt')
    print('Epoch:{}'.format(epoch))
    print('Train loss:{},Train acc:{}'.format(train_loss,train_acc))
    print('Val loss:{},Val acc:{}'.format(valid_loss,valid_acc))
    print(' ')

#第九步：预测结果
model.load_state_dict(torch.load('wordavgmodel.pt'))
import spacy

nlp = spacy.load("en")

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

predict_sentiment('i love this film ')
predict_sentiment('the film is great')
predict_sentiment('the film is not good')
predict_sentiment('the film is not bad')
