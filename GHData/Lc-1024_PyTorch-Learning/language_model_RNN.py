# 语言模型
# 已有一个词，预测下一个单词，模型有LSTM，GRU，RNN三种（RNN很难训练）
# 用LanguageModelingDataset处理数据，再由BPTTIterator转换成迭代器
# 使用Clip防止梯度爆炸，用detach防止数据计算量过大
# 用scheduler减小学习速率


import torchtext
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

SEED = 24
# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(SEED)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
MAX_VOCAB_SIZE = 20000

# Field确定输入的格式
# 并且用自带的datasets处理输入
TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".data\\text8", 
    train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)

# 用自带的BPTT将数据转换成迭代器
VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=torch.device("cuda" if USE_CUDA else "cpu"), 
    bptt_len=32, repeat=False, shuffle=True)
'''
it = iter(train_iter)
batch = next(it)
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,1].data]))
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,1].data]))
for i in range(5):
    batch = next(it)
    print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,2].data]))
    print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,2].data]))
'''


class RNNModel(nn.Module):
    """ 一个简单的循环神经网络"""

    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, n_layers, dropout=0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(RNN, LSTM, GRU)
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization，屏蔽部分神经元，防止过拟合
        '''
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_size, hidden_size, n_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    # hidden层不一定需要随机化，经过几层循环后已经随机化
    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.n_layers, bsz, self.hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((self.n_layers, bsz, self.hidden_size), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.n_layers, bsz, self.hidden_size), requires_grad=requires_grad)

model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 2, dropout=0.5)
if USE_CUDA:
    model = model.cuda()

def evaluate(model, data):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item()*np.multiply(*data.size())
            
    loss = total_loss / total_count
    model.train()
    return loss

# detach将节点截断，防止前面的导数过多影响计算
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# scheduler 可以减小learning_rate
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
GRAD_CLIP = 1.
NUM_EPOCHS = 2

val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        loss.backward()
        # clip将过高和过低的数值截断，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 100 == 0:
            print("epoch", epoch, "iter", i, "loss", loss.item())
    
        if i != 0 and i % 1000 == 0:
            val_loss = evaluate(model, val_iter)
            
            if len(val_losses) == 0 or val_loss < min(val_losses):
                print("best model, val loss: ", val_loss)
                torch.save(model.state_dict(), "language_best_model.th")
            else:
                scheduler.step()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            val_losses.append(val_loss)

        if i == 1000:
            break
        

best_model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 2, dropout=0.5)
if USE_CUDA:
    best_model = best_model.cuda()
best_model.load_state_dict(torch.load("language_best_model.th"))

val_loss = evaluate(best_model, val_iter)
print("perplexity: ", np.exp(val_loss))
test_loss = evaluate(best_model, test_iter)
print("perplexity: ", np.exp(test_loss))
hidden = best_model.init_hidden(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
for i in range(100):
    output, hidden = best_model(input, hidden)
    word_weights = output.squeeze().exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(" ".join(words))