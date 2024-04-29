import collections
import pickle
import nltk
import os
import numpy as np

from pycocotools.coco import COCO
from tqdm import tqdm
from config import hparams
from config import DATA_ROOT
from vocab import load_vocab


def process_caption(cap_dir, vocab_pkl, max_sen_len):
    """
    对caption增加<start>, <end>，截断或填充至固定长度，并将结果保存在{dataset_name}_ids.pkl

    :param max_sen_len:
    :param vocab_pkl:
    :param cap_dir: 划分后的数据集集合根目录，包括train, val, test
    :return:
    """
    vocab = load_vocab(vocab_pkl)

    subsets = ['train', 'val', 'test']
    for subset in subsets:
        ann_file = cap_dir + subset + '_ids.pkl'
        if os.path.exists(ann_file):
            print('*' * 20, os.path.basename(ann_file), 'already exists.', '*' * 20)
            continue

        # {cap_id : {caption: cap , length: len]}
        data_dict = {}
        path = cap_dir + subset + '.json'
        coco = COCO(path)
        ann_ids = list(coco.anns.keys())
        for ann_id in tqdm(ann_ids):
            item_new = collections.OrderedDict()
            cap = coco.anns[ann_id]['caption']
            cap, cap_len = fix_length(cap, vocab, max_sen_len)
            item_new['caption'] = cap
            item_new['length'] = cap_len
            data_dict[ann_id] = item_new

        print('*' * 20, 'save ann_file: ', ann_file, '*' * 20)
        with open(ann_file, 'wb') as f:
            pickle.dump(data_dict, f)


def fix_length(cap, vocab, max_sen_len):
    """
    固定sequence长度: 超过长度的截断，不足长度的填充0
    """
    tokens = nltk.tokenize.word_tokenize(str(cap).lower())
    cap = [vocab('<start>')]
    cap.extend([vocab(token) for token in tokens])
    cap.append(vocab('<end>'))
    cap_len = len(cap)

    cap_tensor = np.zeros(max_sen_len)
    cap_len = (cap_len if cap_len <= max_sen_len else max_sen_len)
    cap_tensor[:cap_len] = np.array(cap[:cap_len])
    # 返回填充后的cap，以及cap不包括填充的实际长度
    return cap_tensor, cap_len


if __name__ == '__main__':
    process_caption(cap_dir=(DATA_ROOT + 'karpathy_split/'), vocab_pkl=hparams.vocab_pkl, max_sen_len=20)
