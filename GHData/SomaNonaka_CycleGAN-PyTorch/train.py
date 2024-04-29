import os
import argparse
import torch
from torch import optim
from tensorboardX import SummaryWriter
from model import Generator, Discriminator
from load_animal_image import load_data
# from load_maps import load_data
from utils import set_requires_grad, ImageBuffer

parser = argparse.ArgumentParser('Train CycleGAN')
parser.add_argument('--channel1', type=int, default=3)
parser.add_argument('--channel2', type=int, default=3)
parser.add_argument('--n_dim', type=int, default=64, help='number of channels of first convolution')
parser.add_argument('--n_res', type=int, default=9, help='number of resnet block in generator')
parser.add_argument('--buffer', type=int, default=50, help='image buffer size for discriminator')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--lambda_', type=float, default=10.0)
parser.add_argument('--lambda_idt', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--data', type=str, default='./train', help='directory that contains training data')
parser.add_argument('--model', type=str, default='./model', help='save trained model to this directory')

opt = parser.parse_args()

if not os.path.exists(opt.model):
    os.makedirs(opt.model)

writer = SummaryWriter('./logs')

# model instantiation
G = Generator(opt.channel1, opt.n_dim, opt.channel2, opt.n_res)
F = Generator(opt.channel2, opt.n_dim, opt.channel1, opt.n_res)
D_X = Discriminator(opt.channel1, opt.n_dim)
D_Y = Discriminator(opt.channel2, opt.n_dim)

# GPU setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G.to(device)
F.to(device)
D_X.to(device)
D_Y.to(device)

# setup optimizers
g_params = list(G.parameters()) + list(F.parameters())
d_params = list(D_X.parameters()) + list(D_Y.parameters())
g_optimizer = optim.Adam(g_params, opt.lr, (opt.beta1, 0.999))
d_optimizer = optim.Adam(d_params, opt.lr, (opt.beta1, 0.999))

# prepare dataset
dataloader = load_data(opt.data, opt.batch, key=['Bengal', 'Bombay'])

# Buffer that contain fake images
buffer_X = ImageBuffer(opt.buffer)
buffer_Y = ImageBuffer(opt.buffer)

for epoch in range(opt.epoch):
    print('epoch: %d' % epoch)

    # change learning rate par 100 epochs
    if (epoch + 1) % 100 == 0:
        opt.lr = opt.lr * 0.1
        for param_group in g_optimizer.param_groups:
            param_group['lr'] = opt.lr
        for param_group in d_optimizer.param_groups:
            param_group['lr'] = opt.lr

    for i, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        # --------train Generator-------#
        X_fake = F(Y)
        X_rec = G(X_fake)
        Y_fake = G(X)
        Y_rec = F(Y_fake)

        set_requires_grad([D_X, D_Y], False)
        g_optimizer.zero_grad()
        G_ad_loss = torch.mean((D_X(X_fake) - 1) ** 2)
        F_ad_loss = torch.mean((D_Y(Y_fake) - 1) ** 2)
        G_cyc_loss = torch.mean((X.detach() - X_rec) ** 2)
        F_cyc_loss = torch.mean((Y.detach() - Y_rec) ** 2)

        if opt.lambda_idt > 0:
            G_idt_loss = torch.mean(torch.abs(G(X) - X)) * opt.lambda_idt
            F_idt_loss = torch.mean(torch.abs(F(Y) - Y)) * opt.lambda_idt
        else:
            G_idt_loss = 0
            F_idt_loss = 0

        gen_loss = G_ad_loss + F_ad_loss + opt.lambda_ * (G_cyc_loss + F_cyc_loss + G_idt_loss + F_idt_loss)
        gen_loss.backward()
        g_optimizer.step()

        # ---------train Discriminator------#
        set_requires_grad([D_X, D_Y], True)
        d_optimizer.zero_grad()

        # train D_X
        loss_dx_real = torch.mean((D_X(X) - 1) ** 2)
        X_fake = buffer_X(X_fake)
        pred_fakeX = D_X(X_fake.detach())
        loss_dx_fake = torch.mean(pred_fakeX ** 2)
        loss_dx = (loss_dx_real + loss_dx_fake) * 0.5
        loss_dx.backward()

        # train D_Y
        loss_dy_real = torch.mean((D_Y(Y) - 1) ** 2)
        Y_fake = buffer_Y(Y_fake)
        pred_fakeY = D_Y(Y_fake.detach())
        loss_dy_fake = torch.mean(pred_fakeY ** 2)
        loss_dy = (loss_dy_real + loss_dy_fake) * 0.5
        loss_dy.backward()

        d_optimizer.step()

        # write logs for tensorboard
        writer.add_scalar('Train/G_loss', gen_loss.cpu().detach().numpy())
        writer.add_scalar('Train/DX_loss', loss_dx.cpu().detach().numpy())
        writer.add_scalar('Train/DY_loss', loss_dy.cpu().detach().numpy())

    if (epoch + 1) % 100 == 0:
        torch.save(G.state_dict(), os.path.join(opt.model, 'G_' + str(epoch + 1) + '.pth'))
        torch.save(F.state_dict(), os.path.join(opt.model, 'F_' + str(epoch + 1) + '.pth'))
        torch.save(D_X.state_dict(), os.path.join(opt.model, 'D_X_' + str(epoch + 1) + '.pth'))
        torch.save(D_Y.state_dict(), os.path.join(opt.model, 'D_Y_' + str(epoch + 1) + '.pth'))
