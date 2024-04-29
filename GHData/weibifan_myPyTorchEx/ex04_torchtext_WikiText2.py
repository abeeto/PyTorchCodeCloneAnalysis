# -*- coding: utf-8 -*-
# weibifan 2022-10-3

"""
WikiText2语言建模数据集是一个超过 1 亿个标记的集合。
它是从维基百科中提取的，并保留了标点符号和实际的字母大小写。它广泛用于涉及长期依赖的应用程序。

怎么抽象描述WikiText2？
一个大的字符串，包括标点符号。
"""
import torch
from torch import nn, Tensor

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import dataset

# 使用方法：先下载，再使用，但是下载过程不提示，不如TorchVision好。
# root – Directory where the datasets are saved. Default: os.path.expanduser(‘~/.torchtext/cache’)
# 默认位置：C:\Users\Wei\.cache\torch\text\datasets
#train_iter = WikiText2(split='train')

# C:\Users\Wei\PycharmProjects\pythonProject3\data\datasets\WikiText2
train_iter = WikiText2(root='data', split='train') #迭代器是单向的

# 从语料构建字典
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz #取整
    data = data[:seq_len * bsz]  #去掉多余的

    #将向量变成矩阵，行数=seq_len/bsz, 列数=bsz
    data = data.view(bsz, seq_len).t().contiguous()  #contiguous 毗邻的
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
