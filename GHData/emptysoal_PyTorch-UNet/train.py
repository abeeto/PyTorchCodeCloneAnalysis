"""
    训练器模块
"""
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import unet
import dataset


# 训练器
class Trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 网络
        self.net = unet.UNet().to(self.device)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.00001)
        self.loss_func = nn.BCELoss()

        self.loader = DataLoader(
            dataset=dataset.Datasets(path),
            batch_size=4,
            shuffle=True,
            num_workers=4
        )

        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f'loaded{model}!')
        else:
            print('No Param!')
        os.makedirs(img_save_path, exist_ok=True)

    # 训练
    def train(self, stop_value):
        epoch = 1
        while True:
            for inputs, labels in tqdm(self.loader, desc=f'Epoch {epoch}/{stop_value}',
                                       ascii=True, total=len(self.loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                out = self.net(inputs)
                loss = self.loss_func(out, labels)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 输入的图像，取第一张
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = out[0]
                # 标签的图像，取第一张
                y = labels[0]
                # 三张图，从第0轴拼接起来，再保存
                img = torch.stack([x, x_, y], 0)
                save_image(img.cpu(), os.path.join(self.img_save_path, f'{epoch}.png'))
            print(f'\nEpoch: {epoch}/{stop_value}, Loss: {loss}')
            torch.save(self.net.state_dict(), self.model)

            # 备份
            if epoch % 50 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print('model_copy is saved !')
            if epoch > stop_value:
                break
            epoch += 1


if __name__ == '__main__':
    t = Trainer(
        r'./self_data/train_images',
        r'./models/model.plt',
        r'./models/model_{}_{}.plt',
        r'./train_img'
    )
    t.train(10)
