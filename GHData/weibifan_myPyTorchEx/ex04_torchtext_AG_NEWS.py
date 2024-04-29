# -*- coding: utf-8 -*-
# weibifan 2022-10-3
# 文本数据的案例：分词，构建字典，构建索引

"""
数据集介绍：AG_NEWS 新闻分类
共4个类别：
1 : World
2 : Sports
3 : Business
4 : Sci/Tec

"""

import torch
from torchtext.datasets import AG_NEWS

# 构建
#C:\Users\Wei\.cache\torch\text\datasets\AG_NEWS
train_iter = iter(AG_NEWS(split='train'))

# 数据集目录 train.csv.promise文件导致失败，删除即可
print(next(train_iter))

print("done")

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print("word - code: ", vocab(['here', 'is', 'an', 'example']))

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

print(text_pipeline('here is the an example'))
print(label_pipeline('10'))


