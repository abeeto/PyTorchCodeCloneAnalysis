import torch
from torch import nn
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torchvision
from tqdm import tqdm
import argparse

os.chdir('/mnt/nfs/work/liaohuang/vae_test')


class Encoder(nn.Module):
    def __init__(self, z_dim, nef=2):
        super().__init__()
        self.z_dim = z_dim
        self.conv0 = nn.Conv2d(1, nef, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # 14x14
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(nef, nef * 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))  # 6x6
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(nef * 2, nef * 4, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))  # 2x2
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(nef * 4, nef * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # 1x1
        self.relu3 = nn.ReLU()
        self.conv_mu = nn.Conv2d(nef * 4, z_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # 1x1
        self.conv_logvar = nn.Conv2d(nef * 4, z_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.fc0 = nn.Linear(nef * 8 * (1 * 1), nef * 8)
        self.fc_mu = nn.Linear(nef * 8, z_dim)
        self.fc_logvar = nn.Linear(nef * 8, z_dim)
        #
        # self.fcs = nn.Sequential(
        #     nn.Linear(28 * 28, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        # )
        # self.fc_mu = nn.Linear(256, z_dim)
        # self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.fc0(x.view(bs, -1))
        return self.fc_mu(x), self.fc_logvar(x)
        #
        # x = self.fcs(x.view(x.shape[0], -1))
        # return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, z_dim, nef=2):
        super().__init__()
        self.z_dim = z_dim
        self.transpose_conv0 = nn.ConvTranspose2d(z_dim, nef * 8, kernel_size=(1, 1), stride=(2, 2),
                                                  padding=(0, 0))  # 1x1
        self.relu0 = nn.ReLU()
        self.transpose_conv1 = nn.ConvTranspose2d(nef * 8, nef * 4, kernel_size=(4, 4), stride=(2, 2),
                                                  padding=(1, 1))  # 6x6
        self.relu1 = nn.ReLU()
        self.transpose_conv2 = nn.ConvTranspose2d(nef * 4, nef * 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
        self.relu2 = nn.ReLU()
        self.transpose_conv3 = nn.ConvTranspose2d(nef * 2, nef, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
        self.relu3 = nn.ReLU()
        self.transpose_conv4 = nn.ConvTranspose2d(nef, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.sigmoid = nn.Sigmoid()
        #
        # self.fcs = nn.Sequential(
        #     nn.Linear(z_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 28 * 28),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = x.view(-1, self.z_dim, 1, 1)
        x = self.transpose_conv0(x)
        x = self.relu0(x)
        x = self.transpose_conv1(x)
        x = self.relu1(x)
        x = self.transpose_conv2(x)
        x = self.relu2(x)
        x = self.transpose_conv3(x)
        x = self.relu3(x)
        x = self.transpose_conv4(x)
        x = self.sigmoid(x)
        return x
        #
        # x = x.view(x.shape[0], -1)
        # return self.fcs(x).view(x.shape[0], 1, 28, 28)


def show_img(img_tensor):
    plt.imshow(img_tensor.cpu().detach().numpy().transpose(1, 2, 0))
    plt.show()


def encoder_img(img_tensor, encoder, use_eps=False):
    img_tensor = img_tensor.view(1, 1, 28, 28).cuda().float()
    mu, logvar = encoder(img_tensor)
    if use_eps:
        eps = torch.randn_like(logvar)
        ret = mu + logvar * eps
    else:
        ret = mu
    return ret


# data_1 = MNIST[3].view(1, 1, 28, 28).cuda().float()
# encoder_img(data_1, ENCODER)

def test_values(decoder):
    Z_dim = 2  # Fixed
    DIM0_FROM = -1
    DIM0_TO = 1
    DIM1_FROM = -1
    DIM1_TO = 1
    N_PRINT = 41
    all_tensors = None
    for i in range(N_PRINT):  # z[0]
        for j in range(N_PRINT):  # z[1]
            v0 = DIM0_FROM + i * (DIM0_TO - DIM0_FROM) / (N_PRINT - 1)
            v1 = DIM1_FROM + i * (DIM1_TO - DIM1_FROM) / (N_PRINT - 1)
            z = torch.zeros((1, Z_dim)).cuda()
            z[0][0] = v0
            z[0][1] = v1
            # z[0][0] = 18.1019
            # z[0][1] = -9.9131
            # print(z)
            # raise ValueError()
            dec_img = decoder(z)
            # show_img(dec_img)
            if all_tensors is not None:
                all_tensors = torch.cat((all_tensors, dec_img))
            else:
                all_tensors = dec_img
    grid = make_grid(all_tensors, nrow=N_PRINT)
    show_img(grid)
    save_image(grid, 'kkk.jpg')
    # NAME = 'epoch100_Zdim256-dim1From-0.5to0.5.jpg'
    # save_image(grid, os.path.join('/mnt/nfs/work/liaohuang/ForkGAN_Research/vae_test/grid', NAME))


#######################

def test_test_data(encoder):
    dim0 = []
    dim1 = []
    targets = []
    DATA = torchvision.datasets.MNIST('data')
    DIM0_MIN = -9999
    DIM0_MAX = 9999
    DIM1_MIN = -9999
    DIM1_MAX = 9999
    for inputs, target in tqdm(zip(DATA.data, DATA.targets)):
        with torch.no_grad():
            encoder = encoder.eval()
            mu, logvar = encoder(inputs.view(1, 1, 28, 28).cuda().float())
            dim0_value = mu[0][0]
            dim1_value = mu[0][1]
            if dim0_value < DIM0_MIN or dim0_value > DIM0_MAX or dim1_value < DIM1_MIN or dim1_value > DIM1_MAX:
                continue
            dim0.append(dim0_value)
            dim1.append(dim1_value)
            targets.append(target)
            inputs.cpu()
            mu.cpu()
            logvar.cpu()
    dim0 = [item.item() for item in dim0]
    dim1 = [item.item() for item in dim1]
    targets = [item.item() for item in targets]
    #
    plt.scatter(dim0, dim1, c=targets, cmap='tab10', s=0.001)
    plt.colorbar()
    plt.savefig('jjj.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, type=str)
    args = parser.parse_args()

    LOADED = torch.load('/mnt/nfs/work/liaohuang/vae_test/experiments/0916-1/model/epoch0000100.pt')
    ENCODER = LOADED['encoder'].cuda()
    DECODER = LOADED['decoder'].cuda()

    if args.mode == 'test_data':
        test_test_data(encoder=ENCODER)
    elif args.mode == 'test_values':
        test_values(decoder=DECODER)
    else:
        raise ValueError('Invalid mode.')
