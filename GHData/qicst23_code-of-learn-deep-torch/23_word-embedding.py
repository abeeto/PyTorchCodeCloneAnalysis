#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:08:13 2018

@author: jangqh
"""
import torch
from torch import nn
from torch.autograd import Variable
##定义词嵌入
embeds = nn.Embedding(2, 5)  ###两个单词，五维
print embeds.weight

embeds.weight.data = torch.ones(2 ,5)
print embeds.weight


###访问第五十个词向量
embeds = nn.Embedding(100, 10)
single_word_embed = embeds(Variable(torch.LongTensor([1])))
print single_word_embed