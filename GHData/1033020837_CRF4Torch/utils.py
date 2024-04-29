import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import *


tokenizer = BertTokenizer.from_pretrained(bert_model)
# 填充、开始与结束标签，输入TOKEN与输出TAG均用这两个标签作为开始与结束
START_SYMBOL, END_SYMBOL = '[CLS]', '[SEP]'
# 所有tag
TAGS = (START_SYMBOL, END_SYMBOL, 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',  )

tag2idx = {tag: idx for idx, tag in enumerate(TAGS)}    # 标签转索引
idx2tag = {idx: tag for idx, tag in enumerate(TAGS)}    # 索引转标签
start_label_id = tag2idx[START_SYMBOL]  # 开始标签的索引
end_label_id = tag2idx[END_SYMBOL]  # 结束标签的索引
tagset_size = len(tag2idx)   # 标签个数

MAX_LEN = max_len - 2   # 需要在句子首尾加上[CLS]和[SEP]符号，所以减去2

# 数据类
class NerDataset(Dataset):
    def __init__(self, f_path):
        # 读取数据，将句子与对应标注存入两个list
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
        sents, tags_li = [], []
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
            tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        self.sents, self.tags_li = sents, tags_li
                
    # 获取一个样本
    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x = tokenizer.convert_tokens_to_ids(words)
        y = [tag2idx[tag] for tag in tags]
        assert len(x) == len(y)
        seqlen = len(tags)

        return x, y, seqlen


    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''
    填充至batch内最大长度
    '''
    f = lambda x: [sample[x] for sample in batch]
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(0, maxlen)
    y = f(1, maxlen)

    f = lambda x, seqlen: [[1] * len(sample[x]) + [0] * (seqlen - len(sample[x])) for sample in batch]
    masks = f(1, max_len)   # 掩码

    f = torch.LongTensor

    return f(x), f(y), seqlens, torch.FloatTensor(masks)
    
