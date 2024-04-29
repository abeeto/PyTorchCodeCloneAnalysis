# * coding:utf-8 *
#@author    :mashagua
#@time      :2019/5/5 7:35
#@File      :lesson_5.py
#@Software  :PyCharm
import torch
from torchtext import data
SEED=12
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
TEXT=data.Field()
LABEL=data.LabelField(dtype=torch.float)


from torchtext import datasets
train_data,test_data=datasets.IMDB.splits(TEXT,LABEL)
import random 
train_data,valid_data=train_data.split(random_state=random.seed(SEED))
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    device=device)

#batch=next(iter(valid_iterator))
#batch.text

import torch.nn as nn
import torch.nn.functional as F
class WordAVEModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,output_size,pad_idx):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embedding_size,padding_idx=pad_idx)
        self.linear=nn.Linear(embedding_size,output_size)
        
    def forward(self,text):
        embedded=self.embed(text)
        #permute重排序，第一个维度在第0个位置，第0个维度在第1个位置，
        embedded=embedded.permute(1,0,2)#[batch_size,seq_len,embedding_size]
        #在每个维度上求平均(),第二个参数，kernel size,第一个位置标示把它全部压扁，第二个位置保持不变
        pooled=F.avg_pool2d(embedded,(embedded.shape[1],1)).squeeze()#[batch_size,1,embedding_size]
        #拿掉中间1这个维度
        return self.linear(pooled)

VOCAB_SIZE=len(TEXT.vocab)
EMBEDDING_SIZE=100
OUTPUT_SIZE=1
PAD_IDX=TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX=TEXT.vocab.stoi[TEXT.unk_token]
model=WordAVEModel(VOCAB_SIZE,
    EMBEDDING_SIZE,
    OUTPUT_SIZE,
    PAD_IDX
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)

pretrained_embedding=TEXT.vocab.vectors
#带_的函数一般标示replace的操作
model.embed.weight.data.copy_(pretrained_embedding)

optimizer=torch.optim.Adam(model.parameters())
#只是针对二分类
crit=nn.BCEWithLogitsLoss()
model=model.to(device)

def binary_accuracy(preds,y):
    rounded_preds=torch.round(torch.sigmoid(preds))
    correct=(rounded_preds==y).float()
    acc=correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, crit):
    epoch_acc, epoch_loss = 0., 0.
    total_len = 0.
    model.train()
    for batch in iterator:
        preds=model(batch.text).squeeze(1)
        loss=crit(preds,batch.label)
        acc=binary_accuracy(preds,batch.label)
        #sgd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()*len(batch.label)
        epoch_acc+=acc.item()*len(batch.label)
        total_len+=len(batch.label)
    return epoch_loss/total_len,epoch_acc/total_len

def evaluate(model, iterator, crit):
    epoch_loss = 0
    epoch_acc = 0
    total_len=0.
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            preds=model(batch.text).squeeze(1)
            loss=crit(preds,batch.label)
            acc=binary_accuracy(preds,batch.label)
            #sgd(不需要)
    
            epoch_loss+=loss.item()*len(batch.label)
            epoch_acc+=acc.item()*len(batch.label)
            total_len+=len(batch.label)
    model.train()
    return epoch_loss/total_len,epoch_acc/total_len

N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
for epoch in range(N_EPOCHS):
    train_loss,train_acc=train(model, train_iterator, optimizer, crit)
    valid_loss,valid_acc=evaluate(model,valid_iterator,crit)
    if valid_acc>best_valid_acc:
        best_valid_acc=valid_acc
        torch.save(model.state_dict(),'wordavg-model.pth')
    print("epoch", epoch, "train loss", train_loss, "train acc", train_acc)
    print("epoch",epoch,"valid loss",valid_loss,"valid acc",valid_acc)


