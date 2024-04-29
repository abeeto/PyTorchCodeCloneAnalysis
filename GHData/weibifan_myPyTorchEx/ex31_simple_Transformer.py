# -*- coding: utf-8 -*-
# weibifan 2022-10-2
#  简版Transformer，①构建文本训练数据。②构建模型，③预训练模型
#  中文版地址：https://pytorch.apachecn.org/#/docs/1.7/27
"""
第1部分：构建文本训练数据
第1步：准备文本材料。包括收集/爬取，清洗等。
①从原始数据源构建文本材料。
②从现有粗粒度数据集构建文本材料。

第2步：文本材料的标注。代价非常高。
①自监督技术，获取标注结果。
②组合现有数据集。
③开发标注工具，找人标注数据。

第3步：构建字典及句子索引序列
① 分词，不同语言各不相同。中文分为按字，按词，是否包括命名实体等等。
② 构建字典。
③ 将token映射为索引。
④ 将句子映射索引的序列。

第4步：构造训练样本，尤其是构建成batch方式。
①对token进行word embedding
②分句。自然分句，固定长度分句。将句子映射为sentence vector
③将多个句子构建成batch，get_batch()
④构建基于batch的迭代器。

第2部分：构建前向模型。先构建模型的类，在初始化成模型对象。
第1步：构建工具函数及工具类
第2步：继承nn.Module类，构建模型类。
第3步：构建模型对象。初始化对象。

注释2：这个Transformer是简版，层数很少，没有ResNet，没有word embedding

第3部分：构建损失函数，后向传播方法，调度方法


第4部分：预训练
第1步：构建预训练函数。
第2步：构建评估函数
第3步：多轮预训练，多轮调用第1步和第2步。
第4步：给出评估结果。调用第2步。

问题：大量的训练数据在哪里加载到GPU的？

问题2：模型的代码在哪里加载到GPU的？

问题3：模型在GPU的计算过程中，怎么调试？

"""

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================
# 第1部分：构建文本训练数据
# ===================================================================
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#加载数据集
# C:\Users\Wei\PycharmProjects\pythonProject3\data\datasets\WikiText2
train_iter = WikiText2(root='data', split='train')  # 迭代器是单向的

# 从Train语料构建字典
tokenizer = get_tokenizer('basic_english')
#按行读取文件中的文本
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# 输出是字符串查表后，对应的整数值，比如Tensor([9,334,997,...])
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size (单次训练用的样本数，通常为2^N，如32、64、128)

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]  # 去掉多余的

    #将向量变成矩阵，行数=seq_len/bsz, 列数=bsz
    data = data.view(bsz, seq_len).t().contiguous()  # contiguous 毗邻的
    return data.to(device)


batch_size = 20  # 对字符串并没有分句，而是简单粗暴的使用20个token
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# source是矩阵。
# 将矩阵按bptt分成块，也就是一个块是一个矩阵（data），维度为 bptt * batch_size
# target则是偏移一个位置，按bptt分成块，并拼接成向量，长度为 bptt * batch_size
# 一次处理 700个（bptt * batch_size）token
bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target

#test
#data, targets = get_batch(train_data, 0)

# ===================================================================
#第2部分：构建前向模型
# ===================================================================

# 第1步：构建工具函数及工具类
# 工具1：构造一个上三角矩阵（upper-triangular matrix），对角线为0
# Transformer decoder中的掩码
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag. """
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


"""
工具2：位置编码，详见相关文档。
输入：输出向量的位数（d_model），最长的字符串数（max_len）
输出：字符串中每个位置对应的向量。
"""

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# 第2步：构建模型类，需要重载3个函数：构造函数，前向函数，参数初始化
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.model_type = 'Transformer'
        # 没有使用nn.Sequence类，后面只能手工处理

        # 第1层：
        self.encoder = nn.Embedding(ntoken, d_model)  #

        # 第2层：
        self.pos_encoder = PositionalEncoding(d_model, dropout)  # 位置编码

        # 第3层：
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # 第4层：
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken) #

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

# 第2步：构建模型对象
# 设置参数，构建模型的对象
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

# ===================================================================
#第3部分：构建损失函数，后向传播方法，调度方法
# ===================================================================
import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# ===================================================================
#第4部分：预训练
# ===================================================================

# 第1步：构建预训练函数
def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200 #打印输出的控制
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    # 获取batch迭代指针。
    num_batches = len(train_data) // bptt #块（chunk）的数量
    # range(start, stop[, step])
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]

        # 处理输入数据data，得到输出output，每次处理一个chunk
        output = model(data, src_mask)

        # 计算损失
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        #防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

# 第1步：构建评估函数
def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        # 枚举每个块（chunk），bptt * batch_size
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


# 第3步：多轮预训练。
best_val_loss = float('inf')
epochs = 3  #循环3次
best_model = None

for epoch in range(1, epochs + 1): # range集合：从start开始（包括）到stop终止（不包括）
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)  #看到了C++的深拷贝

    scheduler.step()

# 输出最终结果
test_loss = evaluate(best_model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
