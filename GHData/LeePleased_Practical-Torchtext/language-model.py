"""
@Translator:    Lee
@BuildTime:     19/02/18
@Software:      Pycharm

用 Torchtext + Pytorch 训练一个语言模型, 度量一段
自然语言的概率: https://web.stanford.edu/~jurafsky/slp3/3.pdf
"""

import torch
import torch.nn as nn
import torch.optim as optim


from torchtext import data
from torchtext.datasets import WikiText2

import tqdm
import spacy
import numpy as np
from spacy.symbols import ORTH


# Spacy 是有一个用于做 NLP 的框架, 这里主要是用到它有
# 关 tokenize 的功能. add_special_case 方法是指对于
# 限制某个字符串 tokenize 分词的方式. 例如下面:
#
#       don't ----> do n't
#
# 可用命令 my_tok.tokenizer.add_special_case(
#           "don't", [{ORTH: "do"}, {ORTH: "n't"}]
#         )
# 来表示. 下面我们主要想限制一些特殊符号的分词.
tokenize = spacy.load("en")

tokenize.tokenizer.add_special_case('<eos>', [{ORTH: "<eos>"}])
tokenize.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
tokenize.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])


def segment(doc):
    """
    用 Spacy 库做分词, 将一段文档切割到若干词汇.
    """

    tokenizer = tokenize.tokenizer
    return [token.text for token in tokenizer(doc)]


# 定义特征域, 表示一段文本, 要求按规则分词并小写化预处理数据集.
TEXT = data.Field(lower=True, tokenize=segment)

# datasets 中存在一些准备好的数据集, 例如下面的 WikiText2, 另外这个
# 命令会在项目目录下自动创建目录 .data 并下载数据 (4.4M), 当然为了能
# 减少读者的疑惑, 在 data 文件夹下 copy 了一份相同的.
train_set, valid_set, test_set = WikiText2.splits(TEXT)

# 下面看看 train/valid/test 分别有多少条数据在其中 (没分词).
print(len(train_set), len(valid_set), len(test_set), end="\n\n")

# 在构建数据集的同时也可以加入预训练的词向量, 当然这里注释掉了.
TEXT.build_vocab(train_set)  # vectors="data/glove.6B.200d"

# 语言模型的核心便是 Iterator, 有子类为 BPTTIterator. 其特殊功能便
# 是将文本连续地切成一段段等长的序列并做 batch, 称为 bbpt, 例如:
#
#   "Machine learning is a field of computer science
#    that gives computers the ability to learn without
#    being explicitly programmed"
#
# 如果规定连续切割长度为 5, 则上述文本会生成一下列表:
#
#   ["Machine", "learning", "is", "a", "field"]
#   ["of", "computer", "science", "that", "gives"]
#   ["computers", "the", "ability", "to", "learn"]
#   ["without", "being", "explicitly", "programmed", EOS]
#
# 由于语言模型是预测下一个单词, 因此上述样本对应的真值标记为:
#
#   ["learning", "is", "a", "field", "of"]
#   ["computer", "science", "that", "gives", "computers"]
#   ["the", "ability", "to", "learn", "without"]
#   ["being", "explicitly", "programmed", EOS, EOS]
#
# 实际训练时可能还需要加 padding, 尤其是在做 n-gram 的情况时.
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train_set, valid_set, test_set),
    batch_size=64, bptt_len=6, repeat=False
)

# 可以看下 iterator 包括那些属性, 使用方法 vars 可以查询.
print(vars(next(iter(train_iter))).keys())

# 可以看看一个 batch 的内容是什么样的, 注意 batch_first=False.
print(next(iter(train_iter)).text[:, :3], end="\n\n")


class RNNLanguageModel(nn.Module):
    """
    之后的工作就是简单的定义一个计算图即可.
    """

    def __init__(self, num_word, embedding_dim, hidden_dim, dropout_rate):
        super(RNNLanguageModel, self).__init__()

        self._dropout = nn.Dropout(dropout_rate)
        self._encoder = nn.Embedding(num_word, embedding_dim)
        self._rnn_cell = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim
        )
        self._decoder = nn.Linear(hidden_dim, num_word)

    def forward(self, input_x):
        embed_x = self._encoder(input_x)
        dropout_x = self._dropout(embed_x)

        rnn_x, _ = self._rnn_cell(dropout_x)
        decode_x = self._decoder(rnn_x)

        # 输出下一个词的预测分布.
        return decode_x


# 定义模型, 优化算法和损失函数.
model = RNNLanguageModel(len(TEXT.vocab), 32, 64, 0.3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # 无监督学习 -> 多分类问题.


# Iterator 被设置 repeat=True, 因此会无穷无尽的训练下去.
for epoch, train_batch in enumerate(train_iter):
    model.train()

    # 预测并根据真值计算损失.
    text, target = train_batch.text, train_batch.target
    predict = model(text)
    train_loss = criterion(
        predict.view(-1, len(TEXT.vocab)), target.view(-1)
    )

    # 清零梯度并做反向更新参数.
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        valid_loss = 0.0
        model.eval()

        with torch.no_grad():
            for valid_batch in tqdm.tqdm(valid_iter, ncols=50):
                text, target = valid_batch.text, valid_batch.target

                predict = model(text)
                batch_loss = criterion(
                    predict.view(-1, len(TEXT.vocab)), target.view(-1)
                )
                valid_loss += batch_loss.item()

        # 每隔 100 轮打印一次模型在 valid 上的损失.
        print("<轮数 {:6d}>, 模型在开发集上总损失为 {};".format(epoch, valid_loss))
    else:
        print("\r[轮数 {:6d}], 模型完成一次训练集 batch 更新;".format(epoch), end="")


def convert_ids_to_tokens(id, vocab, join=None):
    """
    将一组序号 id 映射到一组符号 token 上.
    """

    if isinstance(id, torch.LongTensor):
        id = id.transpose(0, 1).contiguous().view(-1)
    else:
        id = id.transpose().reshape(-1)

    # 这个 vocab 实际上就是 TEXT.vocab.
    utt = [vocab.itos[i] for i in id]

    if join is None:
        return utt
    else:
        return join.join(utt)


# 从 test 中取出一个样本, 打印并测试效果.
tst_sample = next(iter(test_iter))
print(convert_ids_to_tokens(
    np.argmax(tst_sample.data.numpy(), axis=2), TEXT.vocab, join=" "
))
pred_array = model(tst_sample.text).data.numpy()
print(convert_ids_to_tokens(
    np.argmax(pred_array, axis=2), TEXT.vocab, join=" "
))
