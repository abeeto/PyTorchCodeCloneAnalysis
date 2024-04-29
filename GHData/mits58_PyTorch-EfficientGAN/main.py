import os
import urllib.request
import zipfile
import tarfile
import time

import matplotlib.pyplot as plt
from PIL import Image
import PIL
PIL.PILLOW_VERSION = PIL.__version__
import numpy as np

from sklearn.datasets import fetch_openml

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from generator import Generator
from discriminator import Discriminator
from encoder import Encoder


def make_data():
    # フォルダ「data」が存在しない場合は作成する
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    mnist = fetch_openml('mnist_784', version=1, data_home="./data/")  # data_homeは保存先を指定します

    X = mnist.data
    y = mnist.target

    data_dir_path = "./data/img_78/"
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    count7=0
    count8=0
    max_num=200  # 画像は200枚ずつ作成する

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

        # 7と8を200枚ずつ作成したらbreak
        if (count7>=max_num) and (count8>=max_num):
            break


    data_dir_path = "./data/test/"
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    i_start = i+1

    count2=0
    count7=0
    count8=0
    max_num=5  # 画像は5枚ずつ作成する

    for i in range(i_start,len(X)):  # i_startから始める

        # 画像2の作成
        if (y[i] is "2") and (count2<max_num):
            file_path="./data/test/img_2_"+str(count2)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count2+=1

        # 画像7の作成
        if (y[i] is "7") and (count7<max_num):
            file_path="./data/test/img_7_"+str(count7)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count7+=1

        # 画像8の作成
        if (y[i] is "8") and (count8<max_num):
            file_path="./data/test/img_8_"+str(count8)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28*28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)  # 64×64に拡大
            pil_img_f.save(file_path)  # 保存
            count8+=1


    data_dir_path = "./data/img_78_28size/"
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    count7=0
    count8=0
    max_num=200  # 画像は200枚ずつ作成する

    for i in range(len(X)):

        # 画像7の作成
        if (y[i] is "7") and (count7<max_num):
            file_path="./data/img_78_28size/img_7_"+str(count7)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f.save(file_path)  # 保存
            count7+=1

        # 画像8の作成
        if (y[i] is "8") and (count8<max_num):
            file_path="./data/img_78_28size/img_8_"+str(count8)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28*28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f.save(file_path)  # 保存
            count8+=1

        if (count7>=max_num) and (count8>=max_num):
            break


    data_dir_path = "./data/test_28size/"
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    i_start = i+1


    count2=0
    count7=0
    count8=0
    max_num=5  # 画像は5枚ずつ作成する

    for i in range(i_start,len(X)):  # i_startから始める

        # 画像2の作成
        if (y[i] is "2") and (count2<max_num):
            file_path="./data/test_28size/img_2_"+str(count2)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f.save(file_path)  # 保存
            count2+=1

        # 画像7の作成
        if (y[i] is "7") and (count7<max_num):
            file_path="./data/test_28size/img_7_"+str(count7)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28×28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f.save(file_path)  # 保存
            count7+=1

        # 画像8の作成
        if (y[i] is "8") and (count8<max_num):
            file_path="./data/test_28size/img_8_"+str(count8)+".jpg"
            im_f=(X[i].reshape(28, 28))  # 画像を28*28の形に変形
            pil_img_f = Image.fromarray(im_f.astype(np.uint8))  # 画像をPILに
            pil_img_f.save(file_path)  # 保存
            count8+=1


def make_datapath_list():
    train_img_list = list()
    for img_idx in range(200):
        img_path = "./data/img_78_28size/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78_28size/img_8_" + str(img_idx)+'.jpg'
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

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅]白黒

        # 画像の前処理
        img_transformed = self.transform(img)

        return img_transformed


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def anomaly_score(x, fake_image, z_out_real, D, lb=0.1):
    residual_loss = torch.abs(x - fake_image)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    _, x_feat = D(x, z_out_real)
    _, g_feat = D(fake_image, z_out_real)

    discrimination_loss = torch.abs(x_feat - g_feat)
    discrimination_loss = discrimination_loss.view(discrimination_loss.size()[0], -1)
    discrimination_loss = torch.sum(discrimination_loss, dim=1)

    loss_each = (1 - lb) * residual_loss + lb * discrimination_loss
    total_loss = torch.sum(loss_each)

    return total_loss, loss_each, residual_loss


if __name__ == '__main__':
    # Dataset and DetaLoaderの準備
    make_data()
    train_img_list=make_datapath_list()
    mean = (0.5,)
    std = (0.5,)
    train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 動作の確認
    batch_iterator = iter(train_dataloader)  # イテレータに変換
    imges = next(batch_iterator)  # 1番目の要素を取り出す

    # パラメータ
    num_epochs = 1500
    z_dim = 20

    # device, model, optimizer and criterionの定義
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:{}'.format(device))
    G = Generator(z_dim=20).to(device)
    D = Discriminator(z_dim=20).to(device)
    E = Encoder(z_dim=20).to(device)
    lr_g_e = 0.0001
    lr_d = 0.000025
    g_optmizer = torch.optim.Adam(G.parameters(), lr_g_e, [0.5, 0.999])
    d_optmizer = torch.optim.Adam(D.parameters(), lr_d, [0.5, 0.999])
    e_optimizer = torch.optim.Adam(E.parameters(), lr_g_e, [0.5, 0.999])
    criterion = nn.BCEWithLogitsLoss()

    # 重み初期化 and 訓練モード移行
    G.apply(weights_init); D.apply(weights_init); E.apply(weights_init)
    G.train(); D.train(); E.train()
    torch.backends.cudnn.benchmark = True

    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss, epoch_d_loss, epoch_e_loss = 0.0, 0.0, 0.0

        for batch in train_dataloader:
            if batch.size()[0] == 1:
                continue

            bs = batch.size()[0]
            label_real = torch.full((bs, ), 1).to(device)
            label_fake = torch.full((bs, ), 0).to(device)
            batch = batch.to(device)

            # --- Discriminatorの学習 --- #
            z_out_real = E(batch)
            d_out_real, _ = D(batch, z_out_real)

            input_z = torch.randn(bs, z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_fake + d_loss_real

            d_optmizer.zero_grad()
            d_loss.backward()
            d_optmizer.step()

            # --- Generatorの学習 --- #
            input_z = torch.randn(bs, z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optmizer.zero_grad()
            g_loss.backward()
            g_optmizer.step()

            # --- Encoderの学習 --- #
            z_out_real = E(batch)
            d_out_real, _ = D(batch, z_out_real)

            e_loss = criterion(d_out_real.view(-1), label_fake)

            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            # --- 記録 --- #
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_e_loss += e_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('Epoch {:4d}/{} || G_Loss:{:.4f} || D_Loss:{:.4f} || E_Loss:{:.4f} || time:{:.4f} sec.'.format(
                epoch, num_epochs, epoch_g_loss/batch_size,
                epoch_d_loss/batch_size, epoch_e_loss/batch_size,
                t_epoch_finish - t_epoch_start))

    # GANの性能可視化
    G.eval()
    batch_size = 8
    z_dim = 20
    input_z = torch.randn(batch_size, z_dim).to(device)
    fake_images = G(input_z)
    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
    plt.savefig('gan_out.png')

    # 異常検知
    test_img_list = []
    for i in range(0, 5):
        file_path = './data/test_28size/img_2_{}.jpg'.format(i)
        test_img_list.append(file_path)
        file_path = './data/test_28size/img_7_{}.jpg'.format(i)
        test_img_list.append(file_path)
        file_path = './data/test_28size/img_8_{}.jpg'.format(i)
        test_img_list.append(file_path)

    test_dataset = GAN_Img_Dataset(file_list=test_img_list, transform=ImageTransform(mean, std))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True)
    it = iter(test_dataloader)
    imges = next(it)
    x = imges[0:5]
    x = x.to(device)

    z_out_real = E(imges.to(device))
    imges_reconstruct = G(z_out_real)

    loss, loss_each, residual_loss_each = anomaly_score(x, imges_reconstruct, z_out_real, D, lb=0.1)
    loss_each = loss_each.cpu().detach().numpy()
    print('total_loss:', np.round(loss_each, 0))

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

        plt.subplot(2, 5, i + 6)
        plt.imshow(imges_reconstruct[i][0].cpu().detach().numpy(), 'gray')
    plt.savefig('gan_ano_detec_res.png')
