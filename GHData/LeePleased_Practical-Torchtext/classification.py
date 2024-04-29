"""
@Translator:    Lee
@BuildTime:     19/02/18
@Software:      Pycharm

用 Torchtext + Pytorch 完成一个分类任务, 仓库数据不完整, 完整
数据的地址: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

环境: Pytorch 0.4 + Torchtext + Python 3.6
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

import tqdm
import numpy as np
import pandas as pd

# 显示前几行的数据, pandas 用于处理表格型数据.
print(pd.read_csv("data/kaggle/train.csv").head(2))
print(pd.read_csv("data/kaggle/valid.csv").head(2))
print(pd.read_csv("data/kaggle/test.csv").head(2), end="\n\n")

# 特征域, Field 用于定于数据如何被预处理, 如序列化等. 首先
# 是文本数据, 它需要被分词 (sequential + tokenize), 小写 (lower).
TEXT_FIELD = Field(
    sequential=True, tokenize=lambda x: x.split(), lower=True
)
# 其次就是标注 label, 由于源数据是整形的, 因此无需字典 (use_vocab).
LABEL_FIELD = Field(sequential=False, use_vocab=False)

# Dataset 类, 实际上是由多个 Example 组成, Example 的各个
# 属性便是由 Field 组成的. TabularDataset 是 Dataset 的一
# 个子类, 主要用于读取 csv, tsv 和 json 类型的数据.
LABELLED_DATA_FIELD = [
    ("id", None),  # 如果不想要数据中的某个特征, 设为 None 即可.
    ("comment_text", TEXT_FIELD), ("toxic", LABEL_FIELD),
    ("severe_toxic", LABEL_FIELD), ("threat", LABEL_FIELD),
    ("obscene", LABEL_FIELD), ("insult", LABEL_FIELD),
    ("identity_hate", LABEL_FIELD)
]

# splits (classmethod) 方法可以用于同时读取格式相同的数据文件
train_set, valid_set = TabularDataset.splits(
    path="data/kaggle", train="train.csv", validation="valid.csv",
    format="csv", fields=LABELLED_DATA_FIELD, skip_header=True,
)

# 测试数据没有标记, 因此需要额外的格式 Field 来定义.
UNLABELLED_DATA_FILED = [("id", None), ("comment_text", TEXT_FIELD)]

# __call__ 方法可直接读入数据集, 因为格式和 train/dev 不同, 另
# 外如果不是 txt 格式, 那么需要用 skip_header 跳过 csv 的头行.
test_set = TabularDataset(
    path="data/kaggle/test.csv", format="csv", skip_header=True,
    fields=UNLABELLED_DATA_FILED
)

# 通过方法 __dict__  查看对象的属性, 完整数据集要注释掉, 否则刷频...
print(test_set.__dict__.keys())

# 可以通过方法 __getitem__ 获取样本对象 Example, 并输出其定义属性.
print(test_set[0].comment_text)

# 通过训练集构建 word -> id 的映射, 这里用 TEXT_FIELD 的 build_vocab 方法.
TEXT_FIELD.build_vocab(train_set)

# 其中 voacb.freqs 类急速 collections.Counter, 这很便于我们分析数据.
print(TEXT_FIELD.vocab.freqs.most_common(10))

# Iterator 类是要用来造 batch 的对象, 可以把它类比为 Pytorch 里面
# 的 DataLoader 类. BucketBatch 继承自 Iterator, 它非常强大, 可以尽量
# 保持每个 batch 中的数据的长度是尽可能的相同, 避免了 padding 的损失.
train_iter, valid_iter = BucketIterator.splits(
    (train_set, valid_set), batch_sizes=(64, 64),
    # device=-1,  # 如果使用 GPU 需要指定块号, 感觉这个不是那么方便.
    sort_key=lambda x: len(x.comment_text),  # 这是所必须要求的.
    sort_within_batch=False,  # 指每个 batch 内是否需要排序.
    repeat=False,  # 使用 True 表示迭代永无终止, 不适用 epoch 的概念.
)

# 可以试着看下 Iterator 是怎么输出的, 使用 __iter__ 方法.
print(next(train_iter.__iter__()).comment_text)

# 迭代测试集的 batch 时我们是不需要 shuffle 的, 用 Iterator 即可.
test_iter = Iterator(
    test_set, batch_size=64, sort=False,
    sort_within_batch=False, repeat=False
)


class BatchLoaderWrapper:
    """
    在之前的打印中可以看到, Iterator 会返回一个 Batch 对象, 这实际上
    会因为 column 的不同而造成冗余代码 (需要单独处理 test), 因此这里
    需要对 batch 对象进行一个打包, 以提高代码的复用程度.
    """

    def __init__(self, iterator, feature_field, target_field):
        self._iterator = iterator
        self._feature_field = feature_field
        self._target_field = target_field

    def __iter__(self):
        """
        迭代器方法, 实际上是包装了 batch 的 __iter__.
        """

        for batch in self._iterator:
            x = getattr(batch, self._feature_field)

            if self._target_field is not None:
                y = torch.cat(
                    [getattr(batch, feat).unsqueeze(1) for feat in self._target_field],
                    dim=1
                ).float()
            else:
                y = torch.zeros((1,))

            yield (x, y)

    def __len__(self):
        return len(self._iterator)


train_wrapper = BatchLoaderWrapper(
    train_iter, "comment_text",
    [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    ]
)
valid_wrapper = BatchLoaderWrapper(
    valid_iter, "comment_text",
    [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    ]
)
test_wrapper = BatchLoaderWrapper(
    test_iter, "comment_text", None
)

# 现在调用下 next 方法抽取 iterator 会发现特别方便的返回了张量.
print(next(train_wrapper.__iter__()))
print(next(test_wrapper.__iter__()), end="\n\n")


class BiLSTMClassifier(nn.Module):
    """
    之后就是用 Pytorch 定义计算图然后训练模型了.
    """

    def __init__(self, hidden_dim, embedding_dim, dropout_rate):
        super(BiLSTMClassifier, self).__init__()

        self._embed = nn.Embedding(
            len(TEXT_FIELD.vocab), embedding_dim
        )
        self._dropout = nn.Dropout(dropout_rate)
        self._encoder = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim,
            num_layers=1, bidirectional=True
        )
        self._linear = nn.Linear(hidden_dim * 2, 6)

    def forward(self, seq_x):
        embed_x = self._embed(seq_x)
        dropout_x = self._dropout(embed_x)

        hidden, _ = self._encoder(dropout_x)
        feature_x = hidden[-1, :, :]

        pred_x = self._linear(feature_x)
        return pred_x


model = BiLSTMClassifier(32, 16, 0.3)
optimizer = optim.Adam(model.parameters(), 1e-3)
criterion = nn.BCEWithLogitsLoss()

num_epoch = 100
for epoch in range(0, num_epoch + 1):

    model.train()
    total_loss = 0.0

    for x, y in train_wrapper:  # tqdm.tqdm(train_wrapper):
        pred_y = model(x)
        loss = criterion(pred_y, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in valid_wrapper:
            pred_y = model(x)
            loss = criterion(pred_y, y)
            val_loss += loss.item()

    # 打印模型在 train 和 dev 上的总损失.
    print("轮数: {:4d}, 训练集 (train): {:.4f}, 开发集 (valid): {:.4f}"
          ";".format(epoch, total_loss, val_loss))
print("\n训练模型完毕, 下面用模型预测测试集:")

test_pred_list = []
for x, y in test_wrapper:
    pred_x = model(x)
    numpy_x =pred_x.data.numpy()

    prob_x = 1.0 / (1.0 + np.exp(-numpy_x))
    test_pred_list.append(prob_x)

test_array = np.hstack(test_pred_list)
print(test_array)
