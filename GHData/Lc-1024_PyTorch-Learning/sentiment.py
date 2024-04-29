# 训练一个能够做情感分类的模型
# 从datasets.IMDB中获取数据
# 用glove初始化词向量中的参数
# - WordAVGModel 简单的将句子的词向量对应相加，再用linear拟合
# - RNNModel 用两层的LSTM拟合，迫于电脑性能只用一层
# - CNNModel 用三层conv2d拟合，组成一个ModuleList

import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
from torchtext import data
from torchtext import datasets
import random


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# 固定初始化种子
SEED = 24
random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(SEED)


# 定义一些超参数
BATCH_SIZE = 64
EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 20000

NUM_EPOCH_AVG = 16
LEARNING_RATE_AVG = 0.001

NUM_EPOCH_RNN = 6
NUM_LAYERS = 1
HIDDEN_SIZE = 100
DROP_OUT = 0.5
LEARNING_RATE_RNN = 0.001

NUM_EPOCH_CNN = 6
NUM_FILTERS = 100
FILTER_SIZE = [3,4,5]
LEARNING_RATE_CNN = 0.001

LOG_FILE_AVG = "sentiment_wordAVG_model.log"
LOG_FILE_RNN = "sentiment_RNN_model.log"
LOG_FILE_CNN = "sentiment_CNN_model.log"

# spacy是一种特殊的分词方式，除了空格以外还能识别标点符号等
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
# 自动下载IMDB数据集，分成两份，并且用前面的Feilds处理
# 共5w电影评论，评论被分为正负两面（注：文件有200M+，处理速度极慢）
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print("IMDB successfully loaded.")
'''
# 检查一共产生了多少分样本
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[0]))
'''
# 再将train文件分成两部分，以获得valid_data

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# 用自带的build_vocab函数来创建单词表，词向量由预定义的glove提供（注：glove.6B.100d文件有300M+）
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
VOCAB_SIZE = len(TEXT.vocab)
LABEL_SIZE = len(LABEL.vocab)
OUTPUT_SIZE = 1
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
'''
print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)
'''

# 用data.BucketIterator来将所有数据分为合适大小的迭代器
# Bucket会把句子的顺序打乱，但是不会影响单词训练
# 并且保证句子的长度大致相等，防止padding过多
train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE, device=device
)
'''
batch = next(iter(valid_iter))
batch.text
'''


# word averaging 将句子的所有单词的词向量平均化，作为一个词向量用于模型的训练
class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
        super(WordAVGModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.linear = nn.Linear(embedding_size, output_size)
    def forward(self, text):
        embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
        # embedded = embedded.transpose(1, 0) # [batch_size, seq_len, embedding_size]
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)) # [batch_size, 1, embedding_size]
        pooled = pooled.squeeze() # [batch_size, embedding_size]
        result = self.linear(pooled) # [batch_size, 1]
        return result

# 计算模型参数的数量，过多容易过拟合
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 用RNN中的LSTM来拟合
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, hidden_size, bidirectional=False, num_layers=1, drop_out=0.5):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size*num_layers, output_size)
        self.dropout = nn.Dropout(drop_out)
    def forward(self, text):
        embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
        embedded = self.dropout(embedded)
        # 每个句子都是独立的，故不考虑用前一个的hidden
        output, (hidden, cell) = self.lstm(embedded) 
        # output = [sen_len, batch_size, hidden_size * num_directions]
        # hidden = [num_layers * num_directions, batch_size, hidden_size]
        # cell = [num_layers * num_directions, batch_size, hidden_size]

        # 将最后的两层hidden取出作为output
        hidden = torch.cat([hidden[-i] for i in range(NUM_LAYERS)], dim=1)
        hidden = self.dropout(hidden.squeeze())
        return self.linear(hidden)

# 用CNN的conv2d函数拟合
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_filters, filter_size, output_size, pad_idx, drop_out=0.5):
        super(CNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        # self.conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size))
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_size))
            for fs in filter_size
        ])
        self.linear = nn.Linear(num_filters * len(filter_size), output_size)
        self.dropout = nn.Dropout(drop_out)
    def forward(self, text):
        text = text.permute(1, 0) # [batch_size, seq_len]
        embedded = self.embed(text) # [batch_size, seq_len, embedding_size]
        embedded = embedded.unsqueeze(1) # [batch_size, 1, seq_len, embedding_size]
        # conved = F.relu(self.conv(embedded)) # [batch_size, num_filters, seq_len-filter_size+1, 1]
        # conved = conved.squeeze(3) # [batch_size, num_filters, seq_len-filter_size+1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # [batch_size, num_filters, seq_len-filter_size+1]
        # max over time pooling
        # pooled = F.max_pool1d(conved, conved.shape[2]) # [batch_size, num_filters, 1]
        # pooled = pooled.squeeze(2)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        pooled = torch.cat(pooled, dim=1) # [batch_size, 3*num_filters]
        pooled = self.dropout(pooled)

        return self.linear(pooled)


# 预测准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def train(model, train_iter, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.train()
    total_len = 0.
    for batch in train_iter:
        preds = model(batch.text).squeeze() # [batch_size]
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        total_len += 1

    return epoch_loss / total_len, epoch_acc / total_len

def evaluate(model, valid_iter, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.
    for batch in valid_iter:
        with torch.no_grad():
            preds = model(batch.text).squeeze() # [batch_size]
            loss = crit(preds, batch.label)
            acc = binary_accuracy(preds, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            total_len += 1
    
    model.train()

    return epoch_loss / total_len, epoch_acc / total_len

'''
model = WordAVGModel(VOCAB_SIZE, EMBEDDING_SIZE, OUTPUT_SIZE, PAD_IDX).to(device)

# 利用glove中的词向量初始化model的weight，加快训练速度
pre_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pre_embedding)
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_AVG)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
# 用于二分类的数据在零上下的一种损失函数
crit = nn.BCEWithLogitsLoss().to(device)

print("WordAVGModel begins to run.")
fout = open(LOG_FILE_AVG, "w")
best_acc = 0.
to_schedule = 0
for e in range(NUM_EPOCH_AVG):
    train_loss, train_acc = train(model, train_iter, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iter, crit)
    print("Epoch", e, "Train Loss", train_loss, "Train Acc", train_acc)
    print("Epoch", e, "Valid Loss", valid_loss, "Valid Acc", valid_acc)
    fout.write("Epoch: {}, Train Loss: {}, Train Acc: {}\n".format(e, train_loss, train_acc))
    fout.write("Epoch: {}, Valid Loss: {}, Valid Acc: {}\n".format(e, valid_loss, valid_acc))
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), "sentiment_wordAVG_model.pth")
        print("Best model saved.")
        fout.write("Best model saved.")
    else:
        scheduler.step()
        print("learning rate has reduced.")
        fout.write("learning rate has reduced.")
fout.close()

'''
'''
# 下面是RNN模型的训练
model = RNNModel(VOCAB_SIZE, EMBEDDING_SIZE, OUTPUT_SIZE, PAD_IDX, HIDDEN_SIZE, False, NUM_LAYERS, DROP_OUT).to(device)
# 利用glove中的词向量初始化model的weight，加快训练速度
pre_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pre_embedding)
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_RNN)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
# 用于二分类的数据在零上下的一种损失函数
crit = nn.BCEWithLogitsLoss().to(device)

print("RNNModel begins to run.")
fout = open(LOG_FILE_RNN, "w")
best_acc = 0.
to_schedule = 0
for e in range(NUM_EPOCH_RNN):
    train_loss, train_acc = train(model, train_iter, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iter, crit)
    print("Epoch", e, "Train Loss", train_loss, "Train Acc", train_acc)
    print("Epoch", e, "Valid Loss", valid_loss, "Valid Acc", valid_acc)
    fout.write("Epoch: {}, Train Loss: {}, Train Acc: {}\n".format(e, train_loss, train_acc))
    fout.write("Epoch: {}, Valid Loss: {}, Valid Acc: {}\n".format(e, valid_loss, valid_acc))
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), "sentiment_RNN_model.pth")
        print("Best model saved.")
        fout.write("Best model saved.")
    else:
        scheduler.step()
        print("learning rate has reduced.")
        fout.write("learning rate has reduced.")
fout.close()
'''

'''

# 下面是CNN模型的训练
model = CNNModel(VOCAB_SIZE, EMBEDDING_SIZE, NUM_FILTERS, FILTER_SIZE, OUTPUT_SIZE, PAD_IDX, DROP_OUT).to(device)
# 利用glove中的词向量初始化model的weight，加快训练速度
pre_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pre_embedding)
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_CNN)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
# 用于二分类的数据在零上下的一种损失函数
crit = nn.BCEWithLogitsLoss().to(device)

print("CNNModel begins to run.")
fout = open(LOG_FILE_CNN, "w")
best_acc = 0.
to_schedule = 0
for e in range(NUM_EPOCH_CNN):
    train_loss, train_acc = train(model, train_iter, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iter, crit)
    print("Epoch", e, "Train Loss", train_loss, "Train Acc", train_acc)
    print("Epoch", e, "Valid Loss", valid_loss, "Valid Acc", valid_acc)
    fout.write("Epoch: {}, Train Loss: {}, Train Acc: {}\n".format(e, train_loss, train_acc))
    fout.write("Epoch: {}, Valid Loss: {}, Valid Acc: {}\n".format(e, valid_loss, valid_acc))
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), "sentiment_CNN_model.pth")
        print("Best model saved.")
        fout.write("Best model saved.")
    else:
        scheduler.step()
        print("learning rate has reduced.")
        fout.write("learning rate has reduced.")
fout.close()

'''

# 利用训练好的AVG模型对输入的句子进行判断
nlp = spacy.load("en")
def predict_sentiment(sentence):
    tokenize = [tok.text for tok in nlp.tokenizer(sentence)]
    index = [TEXT.vocab.stoi[t] for t in tokenize]
    tensor = torch.LongTensor(index).to(device)
    tensor = tensor.unsqueeze(1)
    pred = torch.sigmoid(model(tensor))
    return pred.item()

model = WordAVGModel(VOCAB_SIZE, EMBEDDING_SIZE, OUTPUT_SIZE, PAD_IDX).to(device)
model.load_state_dict(torch.load("sentiment_wordAVG_model.pth"))

while True:
    sen = input("Please input a command(0 -- stop):")
    if sen == '0':
        break
    print(predict_sentiment(sen), '\n\n')
