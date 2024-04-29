"""
推理
"""

import torch
import os
from transformers import BertTokenizer
from config import *
from models import *


if __name__ == "__main__":

    text = '和往年一样，清明节刚过，我的中学老师就千里迢迢寄来新采制的“雨前茶”，这是一种名叫玉峰云雾茶的绿茶，生长在重庆市郊的玉峰山麓。'
    model = BertLstmCRF().to(device)
    model_save_path = os.path.join(output_dir, 'model.pth')
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
    xx = tokenizer.convert_tokens_to_ids(tokens)
    xx = torch.tensor(xx).unsqueeze(0).to(device)
    masks = torch.tensor([1] * len(tokens)).unsqueeze(0).to(device)

    _, y_hat = model(xx, masks, training=False)
    pred_tags = []
    for tag in y_hat.squeeze()[1:-1]:
        pred_tags.append(idx2tag[tag.item()])

    res_str = [tokens[1:-1], pred_tags]   # res_str[0]为输入句子分词后的列表，res_str[1]为对应标注列表

    for token,tag in zip(res_str[0], res_str[1]):
        print(f'{token}--{tag}',flush=True)


