from torchtext import data,datasets
from torchtext.vocab import GloVe,FastText,CharNGram
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchtext.datasets import IMDB
import sys
from apex import amp

is_cuda = True

TEXT = data.Field(lower=True, batch_first=True,fix_length=40,)
LABEL = data.Field(sequential=False,)

train, test = IMDB.splits(TEXT, LABEL)

print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300),max_size=10000,min_freq=10)
LABEL.build_vocab(train,)

LABEL.vocab.freqs

d = vars(TEXT.vocab)

d.keys()

TEXT.vocab.vectors

len(TEXT.vocab.stoi)

train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32)

train_iter.repeat = False
test_iter.repeat = False


class EmbNet(nn.Module):
    def __init__(self, emb_size, hidden_size1, hidden_size2=400):
        super().__init__()
        self.embedding = nn.Embedding(emb_size, hidden_size1)
        self.fc = nn.Linear(hidden_size2, 3)

    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.fc(embeds)
        return F.log_softmax(out, dim=-1)


model = EmbNet(len(TEXT.vocab.stoi),10).cuda()

optimizer = optim.Adam(model.parameters(),lr=0.001)

model, optimizer = amp.initialize(model, optimizer)

train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32,shuffle=True)
train_iter.repeat = False
test_iter.repeat = False


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, batch in enumerate(data_loader):
        text, target = batch.text, batch.label
        if is_cuda:
            text, target = text.cuda(), target.cuda()

        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, reduction='sum').data.item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

from time import time

ST = time()
for epoch in range(1,10):
    print("Epoch #", epoch, sep="")
    epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

print("Completed in", time() - ST, "seconds.")

# GLOVE

TEXT = data.Field(lower=True, batch_first=True,fix_length=40,)
LABEL = data.Field(sequential=False,)

train, test = IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train,test, vectors=GloVe(name='6B', dim=300),max_size=10000,min_freq=10)
LABEL.build_vocab(train,)

TEXT.vocab.vectors

model = EmbNet(len(TEXT.vocab.stoi),300,12000)
model = model.cuda()
model.embedding.weight.data = TEXT.vocab.vectors.cuda()
model.embedding.weight.requires_grad = False

optimizer = optim.Adam([ param for param in model.parameters() if param.requires_grad == True],lr=0.001)

model, optimizer = amp.initialize(model,optimizer)

train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=64, shuffle=True)
train_iter.repeat = False
test_iter.repeat = False


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, batch in enumerate(data_loader):
        text, target = batch.text, batch.label
        if is_cuda:
            text, target = text.cuda(), target.cuda()

        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, reduction='sum').data.item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

ST=time()
for epoch in range(1,10):
    print("Epoch #", epoch, sep="")
    epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

print("Completed in", time() - ST, "seconds.")
