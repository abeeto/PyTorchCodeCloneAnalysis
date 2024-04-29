import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import DataLoader

from NRE.config import config
from NRE.utils import make_seed, load_pkl
from NRE.process import process
from NRE.dataset import CustomDataset, collate_fn
from NRE import models
from NRE.trainning import train, validate

__Models__ = {
    'CNN': models.CNN,
    'BiLSTM': models.BiLSTM,
    'BiLSTMPro': models.BiLSTMPro
}

parser = argparse.ArgumentParser(description='关系抽取')
parser.add_argument('--model_name', type=str, default='BiLSTMPro', help='model name')
args = parser.parse_args()


if __name__ == '__main__':
    make_seed(config.seed)
    model_name = args.model_name if args.model_name else config.model_name

    if config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', config.gpu_id)
    else:
        device = torch.device('cpu')

    if not os.path.exists(config.out_path):
        process(config.data_path, config.out_path, file_type='csv')

    vocab_path = os.path.join(config.out_path, 'vocab.pkl')
    train_data_path = os.path.join(config.out_path, 'train.pkl')
    test_data_path = os.path.join(config.out_path, 'test.pkl')

    vocab = load_pkl(vocab_path, 'vocab')
    vocab_size = len(vocab.word2id)

    train_dataset = CustomDataset(train_data_path, 'train_data')
    test_dataset = CustomDataset(test_data_path, 'test_data')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn
    )

    model = __Models__[model_name](vocab_size, config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                     factor=config.decay_rate,
                                                     patience=config.decay_patience)
    loss_fn = nn.CrossEntropyLoss()
    print(model)

    best_macro_f1, best_macro_epoch = 0, 1
    best_micro_f1, best_micro_epoch = 0, 1
    best_macro_model, best_micro_model = '',  ''
    print('*****************开始训练****************')

    for epoch in range(1, config.epoch+1):
        train(epoch, device, train_dataloader, model, optimizer, loss_fn, config)
        macro_f1, micro_fi = validate(test_dataloader, model, device, config)
        model_name = model.save_model(epoch=epoch)
        scheduler.step(macro_f1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_macro_epoch = epoch
            best_macro_model = model_name

        if micro_fi > best_micro_f1:
            best_micro_f1 = micro_fi
            best_micro_epoch = epoch
            best_micro_model = model_name

    print("***************模型训练完成************")
    print(f"best marco f1:{best_macro_f1:.4f}", f'in epoch:{best_macro_epoch}', f'save in:{best_macro_model}')
    print(f"best mirco f1:{best_micro_f1:.4f}", f'in epoch:{best_micro_epoch}', f'save in:{best_micro_model}')
