# -*- coding: utf-8 -*-
# weibifan 2022-10-7
# 文本分类模型：嵌入层 + 线性分类器 + AG_NEWS数据集
#  TorchText 及 分类模型 'C:\\Users\\Wei/.cache\\torch\\text\\datasets\\AG_NEWS'
# 第2种方法：继承nn.Module构建模型类。
'''
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
中文版tutorial：https://pytorch.apachecn.org/#/docs/1.7/31

文本分类数据集：
- AG_NEWS,
- SogouNews,
- DBpedia,
- YelpReviewPolarity,
- YelpReviewFull,
- YahooAnswers,
- AmazonReviewPolarity,
- AmazonReviewFull

文本数据比图像数据更复杂
1）需要从数据集构建词典（词汇表），期间还需要分词
2）需要将单词映射为词典的索引
3）需要将索引映射为word vector
4）每个句子的长度是可变的，而图像的大小是固定的。
导致构建数据集的batch指针很困难。

整个程序的结构：
每个模块或者步骤分为2个元步骤：先定义类或函数；然后生成对象并调用函数。

对于常见功能，已经定义好了类，直接调用函数就行。

第0部分：浏览数据集中数据长什么样子


第1部分：文本处理模型的预处理模块
1）配置分词器。需要加载训练数据。
2）构建词汇表。
3）根据词汇表，将token映射为词汇表的索引。

第2部分：构建深度学习模型的类。
1）设置网络基本结构：首先是嵌入层，其次是各种网络层，最后是全连接层。
2）初始化网络参数。
3）设置前向网络。

第3部分：准备训练数据，尤其是batch

第4部分：用模型类构建模型对象，并设置模型对象的参数。

第5部分：训练模型

第5部分：测试模型



'''
import torch
from torch import nn
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('The code runs in ',device)

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#第0部分：浏览数据集中数据长什么样子
'''
train_iter = iter(AG_NEWS(split='train'))
print(next(train_iter))

'''

# 第1部分：文本处理模型的预处理模块
'''
# 分词器，①每种语言各不相同。②n-gram配置不同。③是否区分大小写，命名实体等。
# 对于英文来说：最简单的就是用空格对句子进行分词。
# 对中文来说：需要专门的分词工具。当然可以按字来分。

测试词汇表：
vocab(['here', 'is', 'an', 'example'])
 [475, 21, 30, 5297]
'''
tokenizer = get_tokenizer('basic_english')

# 迭代器是单向的，每用一次，做一次++操作
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 构建词汇表。词汇表从train split获取
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


'''
模型由3层组成：
第1层：将索引变成word embedding
第2层：线性分类器

EmbeddingBag 是针对句子的嵌入，将句子多个单词的向量求平均

模型：
第2层：sent_vec=mean( 句中m个词向量=X * E) ，使用mean转换为一个词向量
第3层：y=fc( batch * W +b) batch由多个sect_vec组成


'''
class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class): #词汇表大小，词向量维度，类别个数
        super(TextClassificationModel, self).__init__()
        # 第1层
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # 第2层
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # 嵌入层权重初始化
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #全连接层权重初始化
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# 第3部分：准备训练数据，尤其是batch

'''
#基于词汇表，将字符串转换为索引
text_pipeline('here is the an example')
>>> [475, 21, 2, 30, 5297]
# 将一个数字组成的字符串转换为数字。 ‘###’ => ###
label_pipeline('10')
>>> 9
'''
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

'''
构造Batch数据
1）无法整除时，需要抛弃部分数据。
2）每个instance之间是否有序关系。如有，batch边界上的序怎么处理。
3）如果instance之间无序关系，是否需要做shuffle？

'''
# 把batch的内容串联在一起，多个句子串联到一起，每个句子由多个单词组成。
# 数据集 多个 [类别，文本]
# batch数据 10个[类别，文本]
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch: #枚举batch中的所有句子和类别
        # 将类别对应数值减一，然后拼接
         label_list.append(label_pipeline(_label))
        # ①句子分词，②找到token在词汇表中的索引，③拼接句子中单词的索引，句子中多个单词的索引是一个list
        # 'here is the an example' 对应的就是  [475, 21, 2, 30, 5297]
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        # ④将多个句子拼接到一起。
         text_list.append(processed_text)
        #将每个句子设置成一个袋子
         offsets.append(processed_text.size(0))
    # 将Python的list数据转换为Tensor
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    # 返回这些数据
    return label_list.to(device), text_list.to(device), offsets.to(device)

# 按batch加载数据
train_iter = AG_NEWS(split='train')
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

# 第4部分：构建模型对象。
train_iter = AG_NEWS(split='train') #一旦被用过，就需要让指针归零
num_class = len(set([label for (label, text) in train_iter]))  #类别个数。
vocab_size = len(vocab) #词汇表中token个数
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

import time

# 第5部分：训练模型

def train(dataloader):
    model.train()  #设置模型为train状态，应该set_train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    # 枚举所有batch
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad() #？？？
        predicted_label = model(text, offsets) # 前向计算一个batch
        loss = criterion(predicted_label, label)  #计算损失函数
        loss.backward() #反向传播，修改权重
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  #normalizing
        optimizer.step()
        # 累加损失
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        # 累加样例数
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            # 引用全局变量 epoch
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():  # 梯度不为0
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss() #损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=LR) # 梯度计算方法
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1) #学习率修改方案
total_accu = None

train_iter, test_iter = AG_NEWS() #获得数据的迭代器

train_dataset = to_map_style_dataset(train_iter) # 将迭代式数据集变成map式数据集
test_dataset = to_map_style_dataset(test_iter)

# 将Train数据集分为2部分，95%的那部分用于训练，5%的那部分用于validate
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# 获得数据集的batch 迭代器
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

# 循环多少次，训练模型
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    # 训练模型
    train(train_dataloader)
    # 验证累模型，
    accu_val = evaluate(valid_dataloader)
    #
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))


ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])