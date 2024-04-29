import torch
from torch import nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm # 进度条可视化库
import numpy as np
import matplotlib.pyplot as plt



def preprocess_img(x):
    x = tfs.ToTensor()(x)  # x (0., 1.)
    return (x - 0.5) / 0.5  # x (-1., 1.)


def deprocess_img(x):  # x (-1., 1.)
    return (x + 1.0) / 2.0  # x (0., 1.)


def discriminator():
    net = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
    )
    return net


def generator(noise_dim):
    net = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 784),
        nn.Tanh(),
    )
    return net


def discriminator_loss(logits_real, logits_fake):  # 判别器的loss
    size = logits_real.shape[0]
    true_labels = torch.ones(size, 1).float()
    false_labels = torch.zeros(size, 1).float()
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss


def generator_loss(logits_fake):  # 生成器的 loss
    size = logits_fake.shape[0]
    true_labels = torch.ones(size, 1).float()
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(logits_fake, true_labels)  # 假图与真图的误差。训练的目的是减小误差，即让假图接近真图。
    return loss


# 使用 adam 来进行训练，beta1 是 0.5, beta2 是 0.999
def get_optimizer(net, LearningRate):
    optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate, betas=(0.5, 0.999))
    return optimizer


def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss,
                noise_size, num_epochs, num_img):
    f, a = plt.subplots(num_img, num_img, figsize=(num_img, num_img))
    plt.ion()  # Turn the interactive mode on, continuously plot

    for epoch in range(num_epochs):
        for iteration, (x, _) in enumerate(train_data):
            bs = x.shape[0]

            # 训练判别网络
            real_data = x.view(bs, -1)  # 真实数据
            logits_real = D_net(real_data)  # 判别网络得分

            rand_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
            fake_images = G_net(rand_noise)  # 生成的假的数据
            logits_fake = D_net(fake_images)  # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake)  # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()  # 优化判别网络

            # 训练生成网络
            rand_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
            fake_images = G_net(rand_noise)  # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake)  # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()  # 优化生成网络

            if iteration % 20 == 0:
                print('Epoch: {:2d} | Iter: {:<4d} | D: {:.4f} | G:{:.4f}'.format(epoch,
                                                                                  iteration,
                                                                                  d_total_error.data.numpy(),
                                                                                  g_error.data.numpy()))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                for i in range(num_img ** 2):
                    a[i // num_img][i % num_img].imshow(np.reshape(imgs_numpy[i], (28, 28)), cmap='gray')
                    a[i // num_img][i % num_img].set_xticks(())
                    a[i // num_img][i % num_img].set_yticks(())
                plt.suptitle('epoch: {} iteration: {}'.format(epoch, iteration))
                plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("cuda能不能用！！！！！！！！！！！！！！！！！！！！！！！！！"+str(torch.cuda.is_available()))
    # 周期
    EPOCH = 5
    # 每次送入网络的图片数量
    BATCH_SIZE = 128
    # 学习率
    LR = 5e-4
    # 噪声的长度
    NOISE_DIM = 96
    # 显示的图像数量
    NUM_IMAGE = 4  # for showing images when training
    # 获取MNIST数据集，如果没有就重新下载
    train_set = MNIST(root='./data/mnist/',
                      train=True,
                      download=True,
                      transform=preprocess_img)
    # 加载数据
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    D = discriminator()
    G = generator(NOISE_DIM)

    D_optim = get_optimizer(D, LR)
    G_optim = get_optimizer(G, LR)

    train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss, NOISE_DIM, EPOCH, NUM_IMAGE)
