from torchtext import data,datasets
from torchtext.vocab import GloVe,FastText,CharNGram
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchtext.datasets import IMDB
import sys
import time
from apex import amp

is_cuda = torch.cuda.is_available()

TEXT = data.Field(lower=True,fix_length=200,batch_first=True)
LABEL = data.Field(sequential=False,)

train, test = IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300),max_size=10000,min_freq=10)
LABEL.build_vocab(train,)

train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=64)
train_iter.repeat = False
test_iter.repeat = False

batch = next(iter(train_iter))

n_vocab = len(TEXT.vocab)
hidden_size = 100


class IMDBCnn(nn.Module):

    def __init__(self, vocab, hidden_size, n_cat, bs=1, kernel_size=3, max_len=200):
        super().__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.e = nn.Embedding(n_vocab, hidden_size)
        self.cnn = nn.Conv1d(max_len, hidden_size, kernel_size)
        self.avg = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(1000, n_cat)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inp):
        bs = inp.size()[0]
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)
        cnn_o = self.cnn(e_out)
        cnn_avg = self.avg(cnn_o)
        cnn_avg = cnn_avg.view(self.bs, -1)
        fc = F.dropout(self.fc(cnn_avg), p=0.2)
        return self.softmax(fc)


model = IMDBCnn(n_vocab, hidden_size, n_cat=3, bs=32, kernel_size=2).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model, optimizer = amp.initialize(model, optimizer)


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

ST=time.time()
for epoch in range(1,10):
    print("Epoch #", epoch, sep="")
    epoch_loss, epoch_accuracy = fit(epoch,model,train_iter,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_iter,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
print("Total Time:", time.time()-ST, "seconds.")