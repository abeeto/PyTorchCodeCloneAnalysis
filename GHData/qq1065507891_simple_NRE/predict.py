import argparse

import pandas as pd
import torch
import os


from NRE.config import config
from NRE.utils import load_pkl, get_result
from NRE.process import split_sentences, build_data

from NRE import models


__Models__ = {
    'CNN': models.CNN,
    'BiLSTM': models.BiLSTM,
    'BiLSTMPro': models.BiLSTMPro
}

parser = argparse.ArgumentParser(description='关系抽取')
parser.add_argument('--model_name', type=str, default='BiLSTMPro', help='model name')
args = parser.parse_args()


if __name__ == '__main__':
    model_name = args.model_name if args.model_name else config.model_name

    if config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', config.gpu_id)
    else:
        device = torch.device('cpu')

    vocab_path = os.path.join(config.out_path, 'vocab.pkl')
    train_data_path = os.path.join(config.out_path, 'train.pkl')

    vocab = load_pkl(vocab_path, 'vocab')
    vocab_size = len(vocab.word2id)

    model = __Models__[model_name](vocab_size, config)
    model.load_model(r'E:\Python\python_file\NLP\信息抽取\my_NRE\checkpoints\BiLSTM_epoch7_0804_21_31_26.pth')
    print(model)

    print('*****************开始预测****************')

    while True:
        text = input('input:')
        data = text.split('#')
        entity1 = data[1]
        entity2 = data[3]
        head_index = data[0].index(entity1)
        tail_index = data[0].index(entity2)
        data.insert(3, head_index)
        data.insert(6, tail_index)
        data.insert(1, 0)
        columns = ['sentence', 'relation', 'head', 'head_type', 'head_offset', 'tail', 'tail_type', 'tail_offset']
        dict_data = {
                c: d for c, d in zip(columns, data)
            }
        raw_data = pd.DataFrame(dict_data,
                                columns=columns,
                                index=[0])

        new_text = split_sentences(raw_data)
        sents, head_pos, tail_pos, mask_pos = build_data(new_text, vocab)
        x = [torch.LongTensor(sents), torch.LongTensor(head_pos), torch.LongTensor(tail_pos), torch.LongTensor(mask_pos)]
        model.eval()

        with torch.no_grad():
            y_pred = model(x)
            y_pred = y_pred.argmax(dim=-1)
            result = get_result(entity1, entity2, y_pred.numpy()[0])
            print(result)
