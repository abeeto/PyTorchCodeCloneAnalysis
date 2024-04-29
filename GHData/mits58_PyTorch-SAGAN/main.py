import json
import os
import tarfile
import time
import urllib.request
import zipfile

from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
import PIL
# PIL.PILLOW_VERSION = PIL.__version__
from torchvision import transforms

from traceback import print_exception
import sys
import io
import requests
import psutil
from functools import wraps

from generator import Generator
from discriminator import Discriminator
import setting

# PILが7.0.0だとPILLOW_VERSIONが存在しないため
# Errorになるので、無理やり代入（あまり良くないかも）


def make_datapath_list():
    train_img_list = list()

    for idx in range(400):
        img_path = "./data/img_78/img_7_{0}.jpg".format(idx)
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_{0}.jpg".format(idx)
        train_img_list.append(img_path)

    return train_img_list


class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, ind):
        img_path = self.file_list[ind]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        return img_transformed

def make_data():
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    mnist = fetch_openml('mnist_784', version=1, data_home="./data/")
    X = mnist.data
    y = mnist.target

    plt.imshow(X[0].reshape(28, 28), cmap='gray')

    data_dir_path = "./data/img_78/"
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    count7=0
    count8=0
    max_num=400

    for i in range(len(X)):
        # 画像7の作成
        if (y[i] is "7") and (count7<max_num):
            file_path="./data/img_78/img_7_"+str(count7)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count7+=1

        # 画像8の作成
        if (y[i] is "8") and (count8<max_num):
            file_path="./data/img_78/img_8_"+str(count8)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28*28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count8+=1


def notify(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        output = io.StringIO()
        try:
            func(*args, **kwargs)
            send_slack_msg()
        except Exception:
            exp_type, exp_val, tb = sys.exc_info()
            print_exception(exp_type, exp_val, tb, limit=1, file=output)
            send_slack_msg(output.getvalue(), success=False)
            raise

    return wrapper


def send_slack_msg(msg="", success=True):
    web_hock_url = setting.web_hock_url
    header = {'Content-type': 'application/json'}

    proc_name = " ".join(psutil.Process().cmdline()[1:])

    if success:
        msg = f"Your job {proc_name} succeeded! :kissing_heart:\n{msg}"
    else:
        msg = f"Sorry, your job {proc_name} failed! :zany_face:\n```{msg}```"

    data = {'text': msg}
    response = requests.post(web_hock_url, headers=header, json=data)
    if response.ok:
        print('Sent a message!')
    else:
        print("Unable to send a message...")


@notify
def main():
    # データセットの準備
    make_data()
    train_img_list = make_datapath_list()
    mean, std = (0.5, ), (0.5, )
    train_dataset = GAN_Img_Dataset(train_img_list, ImageTransform(mean, std))
    batch_size = 64
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size, shuffle=True)

    # モデルの定義と重み初期化
    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(image_size=64)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G.apply(weights_init)
    D.apply(weights_init)

    # decide device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:\t", device)
    G.to(device)
    D.to(device)

    # define optimizer
    g_lr, d_lr = 0.0001, 0.0004
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [0, 0.9])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [0, 0.9])

    # パラメタ
    z_dim = 20  # 乱数の次元

    G.train()
    D.train()
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(train_dataloader.dataset)
    iteration = 1
    logs = []

    # 学習 (300 Epochs)
    for epoch in range(300):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch in train_dataloader:
            # バッチサイズ確認
            if batch.size()[0] == 1:
                continue

            # ラベルの準備
            batch = batch.to(device)
            batch_num = batch.size()[0]
            label_real = torch.full((batch_num, ), 1).to(device)
            label_fake = torch.full((batch_num, ), 0).to(device)

            # --- Discriminatorの学習 --- #
            # 真の画像を判定
            d_out_real, _, _ = D(batch)

            # 偽の画像を生成・判定
            input_z = torch.randn(batch_num, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images, _, _ = G(input_z)
            d_out_fake, _, _ = D(fake_images)

            # 損失を計算・パラメータ更新
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # --- Generatorの学習 --- #
            input_z = torch.randn(batch_num, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images, _, _ = G(input_z)
            d_out_fake, _, _ = D(fake_images)

            # 損失を計算・パラメータ更新
            g_loss = -d_out_fake.mean()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('epoch {:3d}/300 || D_Loss: {:.4f} || G_Loss: {:.4f} || time: {:.4f} sec.'.format(
            epoch, epoch_d_loss / batch_size, epoch_g_loss / batch_size,
            t_epoch_finish - t_epoch_start
        ))

    # --- 画像生成・可視化する --- #
    test_size = 5   # 可視化する個数
    input_z = torch.randn(test_size, z_dim)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    G.eval()
    fake_images, at_map1, at_map2 = G(input_z.to(device))

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        # top: fake image
        plt.subplot(2, 5, i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

        # middle: atmap1
        plt.subplot(2, 5, i+6)
        am = at_map1[i].view(16, 16, 16, 16)
        am = am[7][7]
        plt.imshow(am.cpu().detach().numpy(), 'Reds')

    plt.savefig('visualization.png')


if __name__ == '__main__':
    main()
