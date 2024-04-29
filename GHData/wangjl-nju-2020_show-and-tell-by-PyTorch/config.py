"""
模型配置文件
"""
import torch

from argparse import Namespace

DATA_ROOT = '/home/user_data55/wangjl/cocodata/'
WORK_ROOT = '/home/wangjl/work_dir/nic/'  # 不要使用~

DEVICE = ('cuda:2' if torch.cuda.is_available() else 'cpu')

args = {

    'fixed_seed': 0,

    # 模型参数
    'batch_size': 100,
    'num_workers': 8,
    'fea_dim': 2048,
    'hid_dim': 512,
    'embed_dim': 512,
    'max_sen_len': 20,
    'max_epoch': 50,
    'beam_num': 3,
    'test_beam_num': 3,

    # 优化器参数
    'lr': 4e-4,
    'grad_clip': 0.1,

    # 微调参数
    'fine_tune': False,
    'pretrained_epoch': 20,
    'ft_encoder_lr': 1e-5,
    'ft_decoder_lr': 4e-4,

    # 其他
    'train_img_dir': DATA_ROOT + 'train2014',
    'val_img_dir': DATA_ROOT + 'val2014',
    'img_dirs': [DATA_ROOT + 'train2014', DATA_ROOT + 'val2014'],
    'vocab_pkl': DATA_ROOT + 'karpathy_split/vocab.pkl',
    'train_cap': DATA_ROOT + 'karpathy_split/train.json',
    'val_cap': DATA_ROOT + 'karpathy_split/val.json',
    'test_cap': DATA_ROOT + 'karpathy_split/test.json',
    'train_pkl': DATA_ROOT + 'karpathy_split/train_ids.pkl',
    'cap': DATA_ROOT + 'karpathy_split/{}.json',
    'test_path': WORK_ROOT + 'test.json',
    'val_path': WORK_ROOT + 'val.json',
    'save_path': WORK_ROOT + 'best_model.ckpt',
}

hparams = Namespace(**args)
