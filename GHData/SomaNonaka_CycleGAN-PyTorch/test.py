import os
import argparse
import torch
from model import Generator, Discriminator
from load_animal_image import load_data
# from load_maps import load_data
from scipy.misc import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--channel1', type=int, default=3)
parser.add_argument('--channel2', type=int, default=3)
parser.add_argument('--n_dim', type=int, default=64, help='number of channels of first convolution')
parser.add_argument('--n_res', type=int, default=9, help='number of resnet block in generator')
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--data', type=str, default='./test', help='directory that contains test data')
parser.add_argument('--model', type=str, default='./model', help='load trained model from this directory')
parser.add_argument('--sample', type=str, default='./sample', help='save sample images to this directory')

opt = parser.parse_args()

if not os.path.exists(opt.sample):
    os.makedirs(opt.sample)

# model instantiation
G = Generator(opt.channel1, opt.n_dim, opt.channel2, opt.n_res)
F = Generator(opt.channel2, opt.n_dim, opt.channel1, opt.n_res)
D_X = Discriminator(opt.channel1, opt.n_dim)
D_Y = Discriminator(opt.channel2, opt.n_dim)

# load weights
G.load_state_dict(torch.load(os.path.join(opt.model, 'G_200.pth')))
F.load_state_dict(torch.load(os.path.join(opt.model, 'F_200.pth')))
D_X.load_state_dict(torch.load(os.path.join(opt.model, 'D_X_200.pth')))
D_Y.load_state_dict(torch.load(os.path.join(opt.model, 'D_Y_200.pth')))

# GPU setting
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
G.to(device)
F.to(device)
D_X.to(device)
D_Y.to(device)

# prepare dataset
dataloader = load_data(opt.data, opt.batch, key=['Bengal', 'Bombay'])

# convert images and save
for i, (X, Y) in enumerate(dataloader):
    X, Y = X.to(device), Y.to(device)
    fake_Y = G(X)
    fake_X = F(Y)

    real_X = X.cpu().detach().numpy().transpose(0, 2, 3, 1)
    real_Y = Y.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fake_X = fake_X.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fake_Y = fake_Y.cpu().detach().numpy().transpose(0, 2, 3, 1)

    for j in range(opt.batch):
        imsave(os.path.join(opt.sample, 'real_X' + str(i) + '_' + str(j) + '.png'), real_X[j])
        imsave(os.path.join(opt.sample, 'real_Y' + str(i) + '_' + str(j) + '.png'), real_Y[j])
        imsave(os.path.join(opt.sample, 'fake_X' + str(i) + '_' + str(j) + '.png'), fake_X[j])
        imsave(os.path.join(opt.sample, 'fake_Y' + str(i) + '_' + str(j) + '.png'), fake_Y[j])
