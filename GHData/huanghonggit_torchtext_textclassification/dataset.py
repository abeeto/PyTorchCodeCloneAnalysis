import re
from torchtext import data
import jieba
import logging
jieba.setLogLevel(logging.INFO)

# regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
#
#
# def word_cut(text):
#     text = regex.sub(' ', text)
#     return [word for word in jieba.cut(text) if word.strip()]


# def get_dataset(path, text_field, label_field):
#     # text_field.tokenize = word_cut
#     train, dev = data.TabularDataset.splits(
#         path=path, format='csv', skip_header=True,
#         train='train.csv', validation='test.csv',
#         fields=[
#             # ('index', None),
#             ('label', label_field),
#             ('text', text_field)
#         ]
#     )
#     return train, dev
from torchtext import data


def get_dataset(path, text_field, label_field):
    train, test = data.TabularDataset.splits(
        path=path, format='csv', skip_header=True,
        train='train.csv', validation='test.csv',
        fields=[
            ('text', text_field),
            ('label', label_field)
        ]
    )
    return train, test

