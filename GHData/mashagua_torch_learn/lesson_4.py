# * coding:utf-8 *
# @author    :mashagua
# @time      :2019/5/2 15:06
# @File      :lesson_4.py
# @Software  :PyCharm
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors
import torch
import random
import numpy as np
USE_CUDA = torch.cuda.is_available()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 50000
HIDDEN_SIZE = 100

TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".", train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)

TEXT.vocab.stoi['july']

device = torch.device("cuda" if USE_CUDA else "cpu")
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=device, bptt_len=32, repeat=False, shuffle=True)

it = iter(train_iter)
batch = next(it)
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:, 2].data]))
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:, 2].data]))


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        # 从x传入到
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.Linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, text, hidden):
        # text:seq_len,batch_size
        emb = self.embed(text)
        output, hidden = self.lstm(emb, hidden)
        # 取两维变量
        out_vocab = self.Linear(output.view(-1, output.shape[2]))
        out_vocab = out_vocab.view(output.size(
            0), output.size(1), out_vocab.size(-1))
        return out_vocab, hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())

        return (
            weight.new_zeros(
                (1, bsz, self.hidden_size), requires_grad=True), weight.new_zeros(
                (1, bsz, self.hidden_size), requires_grad=True))


model = RNNModel(
    vocab_size=len(
        TEXT.vocab),
    embed_size=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    model = model.cuda()
next(model.parameters())
#####
loss_fn = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#torch中可以降一部分
scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,0.5)



NUM_EPOCHS = 2
GRAD_CLIP = 5.0
VOCAB_SIZE = len(TEXT.vocab)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        # 若有两个，那么把每一个都做同样的操作
        return tuple(repackage_hidden(v) for v in h)

def evaluate(model,data):
    model.eval()
    total_loss=0.
    total_count=0.

    it = iter(data)
    #所有的都只是在做预测
    with torch.no_grad():
        #不需要backprob
        hidden = model.init_hidden(BATCH_SIZE,requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            # trick的地方，若把所有的hidden在pytorch中都保存
            # 那么很容易爆掉，所以，把当前的hidden的detach掉，
            # 因此每次计算都从当前开始，并且会保留历史的值
            output, hidden = model(data, hidden)
            # batch_size*target_class_dim
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            #loss是被平均过，所以把所有的loss加起来，data.size是个tuple,把这个tuple拆开
            total_loss=loss.item()*np.multiply(*data.size())
            total_count=np.multiply(*data.size())
    loss=total_loss/total_count
    #返回train的状态
    model.train()
    return loss

val_losses=[]
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        # trick的地方，若把所有的hidden在pytorch中都保存
        # 那么很容易爆掉，所以，把当前的hidden的detach掉，
        # 因此每次计算都从当前开始，并且会保留历史的值
        output, hidden = model(data, hidden)
        # batch_size*target_class_dim
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        # 把梯度都norm到一个区间下
        torch.nn.utils.clip_grad_norm(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if i % 100 == 0:
            print('epoch:',epoch,'iteration',i,"loss", loss.item())
        
        if i%1000==0:
            val_loss=evaluate(model,val_iter)
            
            if len(val_losses)==0 or val_loss<min(val_losses):
                torch.save(model.state_dict(),'lm.pth')
                print("best model saved!!!!")
        else:
          scheduler.step()
        val_losses.append(val_loss)
###load model

best_model = RNNModel(
        vocab_size=len(TEXT.vocab),
        embed_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    best_model = best_model.cuda()

best_model.load_state_dict(torch.load('lm.pth'))
#拿batchsize==1的一个hidden_state
hidden=best_model.init_hidden(1)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
input=torch.randint(VOCAB_SIZE,(1,1),dtype=torch.long).to(device)
words=[]
for i in range(100):
    #run forward pass
    output,hidden=best_model(input,hidden)
    ##squeeze是把为1的部分全部扔掉，100*1*1变成100维，加exp是让分布的差距更大
    #logits exp
    word_weights=output.squeeze().exp().cpu()
    #multinomial sampling
    word_idx=torch.multinomial(word_weights,1)[0]
    #fill in the current predicted word to the current input
    input.fill_(word_idx)
    word=TEXT.vocab.itos[word_idx]
    words.append(word)
print(" ".join(words))





            
