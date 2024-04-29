# -*- coding: utf-8 -*-
# weibifan 2022-10-12
#  文本处理的标准层：嵌入层（embedding），分为词向量，句子向量。
#  比图像复杂原因：①需要分词，图像是像素。②将token转换为数值。③句子长度可变。
'''
嵌入（nn.Embedding ）包括word vector和sentence vector。默认是词向量。

第1类：词向量。nn.Embedding 是一个查找表，每行为一个词的词向量。
形状是(num_words, embedding_dim)，num_words表示词汇表大小

如果单词的表征是one hot，比如dog的[0,0,0,1,0,0,0,0,0,0]，词汇表中有10个单词
word_vec = x * E

如果单词的表征是index，比如dog的index是3，
word_vec = E[3]

词嵌入表，可以当做权重，通过学习获得，也可以随机初始化。

第2类：句子向量。对于NLP，通常处理句子，每个句子由可变长度的单词构成，需要构建sentence vector
nn.EmbeddingBag  用offset将一个序列划分为多个袋子bag，每个袋子内向量进行sum，mean等操作，形成一个向量。
应用场景：每个instance是一个句子，此时需要将一个句子作为一个袋子，袋子中每个单词对应一个词向量，然后取平均。

NLP一般都是batch方式。也就是word vector的batch，和sentence vector的batch。

'''

import torch
from torch import nn

if __name__ == '__main__':
    #构建对象，并进行随机初始化
    embedding = nn.Embedding(10, 5)  # 10个词，每个词用2维词向量表示
    print(embedding.weight.size())
    print(embedding.weight)

    # 构建batch，也就是索引
    input = torch.arange(0, 6).view(3, 2).long()  # 3个句子，每句子有2个词
    print(input)
    input = torch.autograd.Variable(input) #这句在干啥？
    print(input)

    # 依据batch索引(形状为（3,2）），取出词向量。
    output = embedding(input) # 根据索引，取出词向量
    print(output)
    print(output.size())

    print('----------------sum---------------------------------------------------')
    # 对每个袋子中所有向量做相加处理
    embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
    print(embedding_sum.weight)

    #offsets会分为两个bags：input[0:2]和input[4:]
    input_sum = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
    offsets = torch.LongTensor([0, 2])

    output_sum = embedding_sum(input_sum, offsets)
    print(output_sum)
    print(output_sum.size())


