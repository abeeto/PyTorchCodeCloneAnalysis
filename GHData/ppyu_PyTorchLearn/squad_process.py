# -*- coding: utf-8 -*-
"""
@File   : squad_process.py
@Author : Pengy
@Date   : 2020/10/12
@Description : Input your description here ... 
"""
from transformers import SquadV2Processor, squad_convert_examples_to_features
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    data_dir = './datasets/squad/'

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    processor = SquadV2Processor()
    examples = processor.get_dev_examples(data_dir)

    features = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=32,
        is_training=True,
    )

    print(features.size)