import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from data_loaders import get_train_loader, get_val_loader
from metrics import coco_eval
from models import CNN, RNN
from config import DEVICE
from config import hparams


class Trainer:
    """
    训练
    """

    def __init__(self, _hparams):
        utils.set_seed(_hparams.fixed_seed)

        self.train_loader = get_train_loader(_hparams)
        self.val_loader = get_val_loader(_hparams)
        self.encoder = CNN().to(DEVICE)
        self.decoder = RNN(fea_dim=_hparams.fea_dim,
                           embed_dim=_hparams.embed_dim,
                           hid_dim=_hparams.hid_dim,
                           max_sen_len=_hparams.max_sen_len,
                           vocab_pkl=_hparams.vocab_pkl).to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.get_params(), lr=_hparams.lr)
        self.writer = SummaryWriter()

        self.max_sen_len = _hparams.max_sen_len
        self.val_cap = _hparams.val_cap
        self.ft_encoder_lr = _hparams.ft_encoder_lr
        self.ft_decoder_lr = _hparams.ft_decoder_lr
        self.best_CIDEr = 0

    def fine_tune_encoder(self, fine_tune_epochs, val_interval, save_path, val_path):
        print('*' * 20, 'fine tune encoder for', fine_tune_epochs, 'epochs', '*' * 20)
        self.encoder.fine_tune()
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.ft_encoder_lr},
            {'params': self.decoder.parameters(), 'lr': self.ft_decoder_lr},
        ])
        self.training(fine_tune_epochs, val_interval, save_path, val_path)
        self.encoder.froze()
        print('*' * 20, 'fine tune encoder complete', '*' * 20)

    def get_params(self):
        """
        模型需要优化的全部参数，此处encoder暂时设计不用训练，故不加参数

        :return:
        """
        return list(self.decoder.parameters())

    def training(self, max_epochs, val_interval, save_path, val_path):
        """
        训练

        :param val_path: 保存验证过程生成句子的路径
        :param save_path: 保存模型的地址
        :param val_interval: 验证的间隔
        :param max_epochs: 最大训练的轮次
        :return:
        """
        print('*' * 20, 'train', '*' * 20)
        for epoch in range(max_epochs):
            self.set_train()

            epoch_loss = 0
            epoch_steps = len(self.train_loader)
            for step, (img, cap, cap_len) in tqdm(enumerate(self.train_loader)):
                # batch_size * 3 * 224 * 224
                img = img.to(DEVICE)
                cap = cap.to(DEVICE)

                self.optimizer.zero_grad()

                features = self.encoder.forward(img)
                outputs = self.decoder.forward(features, cap)

                outputs = pack_padded_sequence(outputs, cap_len - 1, batch_first=True)[0]
                targets = pack_padded_sequence(cap[:, 1:], cap_len - 1, batch_first=True)[0]
                train_loss = self.loss_fn(outputs, targets)
                epoch_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()

            epoch_loss /= epoch_steps
            self.writer.add_scalar('epoch_loss', epoch_loss, epoch)
            print('epoch_loss: {}, epoch: {}'.format(epoch_loss, epoch))
            if (epoch + 1) % val_interval == 0:
                CIDEr = self.validating(epoch, val_path)
                if self.best_CIDEr <= CIDEr:
                    self.best_CIDEr = CIDEr
                    self.save_model(save_path, epoch)

    def save_model(self, save_path, train_epoch):
        """
        保存最好的模型

        :param save_path: 保存模型文件的地址
        :param train_epoch: 当前训练的轮次
        :return:
        """
        model_state_dict = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'tran_epoch': train_epoch,
        }
        print('*' * 20, 'save model to: ', save_path, '*' * 20)
        torch.save(model_state_dict, save_path)

    def validating(self, train_epoch, val_path):
        """
        验证

        :param val_path: 保存验证过程生成句子的路径
        :param train_epoch: 当前训练的epoch
        :return:
        """
        print('*' * 20, 'validate', '*' * 20)
        self.set_eval()
        sen_json = []
        with torch.no_grad():
            for val_step, (img, img_id) in tqdm(enumerate(self.val_loader)):
                img = img.to(DEVICE)
                features = self.encoder.forward(img)
                sens, _ = self.decoder.sample(features)
                sen_json.append({'image_id': int(img_id), 'caption': sens[0]})

        with open(val_path, 'w') as f:
            json.dump(sen_json, f)

        result = coco_eval(self.val_cap, val_path)
        scores = {}
        for metric, score in result:
            scores[metric] = score
            self.writer.add_scalar(metric, score, train_epoch)

        return scores['CIDEr']

    def set_train(self):
        self.encoder.train()
        self.decoder.train()

    def set_eval(self):
        self.encoder.eval()
        self.decoder.eval()


if __name__ == '__main__':
    trainer = Trainer(hparams)
    trainer.validating(0, hparams.val_path)
