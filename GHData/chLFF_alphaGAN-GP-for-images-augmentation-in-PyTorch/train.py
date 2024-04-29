import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from PIL import Image as img
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type=str, default='./data/')
parser.add_argument('--savePath', type=str, default='default')
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--ctn', type=int, default=0,
                    help='continue form breakpoint')
parser.add_argument('--l1', type=float, default=10)
parser.add_argument('--penalty', type=float, default=10)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.9)
parser.add_argument('--b2', type=float, default=0.999)

opt = parser.parse_args()

savePath = opt.savePath
dataPath = opt.dataPath
num_epoch = opt.epochs
ctn = opt.ctn
weight_of_l1 = opt.l1
weight_of_penalty = opt.penalty
learning_rate = opt.lr
b1 = opt.b1
b2 = opt.b2

batch_size = 32
z_dimension = 256


def make_save_path():
    assert not os.path.exists('./saved_images/' + savePath + '/')
    os.makedirs('./saved_images/' + savePath + '/')
    os.makedirs('./dict/' + savePath + '/')


def to_img(x):
    out = (x + 1)/2
    return out


def data_gen(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images


def calc_gradient_penalty(model, x, x_gen, w=10):
    """WGAN-GP gradient penalty"""
    assert x.size() == x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha = torch.cuda.FloatTensor(*alpha_size).uniform_()
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+1e-15).sum(-1).sqrt()

    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(
        model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty


tran = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(dataPath, loader=img.open, transform=tran)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
del dataset

G = Generator(z_size=z_dimension).cuda()
CD = Code_Discriminator(code_size=z_dimension).cuda()
D = Discriminator().cuda()
E = Encoder(out_class=z_dimension).cuda()

# continue from breakpoint
if ctn:
    G.load_state_dict(torch.load('./dict/' + savePath + '/' +
                                 savePath + '_' + str(ctn-1) + '_G.pth'))
    CD.load_state_dict(torch.load(
        './dict/' + savePath + '/' + savePath + '_CD.pth'))
    D.load_state_dict(torch.load(
        './dict/' + savePath + '/' + savePath + '_D.pth'))
    E.load_state_dict(torch.load(
        './dict/' + savePath + '/' + savePath + '_E.pth'))
else:
    make_save_path()

g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
cd_optimizer = torch.optim.Adam(CD.parameters(), lr=learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
e_optimizer = torch.optim.Adam(E.parameters(), lr=learning_rate)

criterion_l1 = nn.L1Loss()

g_iter = 1
d_iter = 1
cd_iter = 1
e_iter = 1

gen_img = data_gen(dataloader)

writer = SummaryWriter(comment=savePath)

for epoch in range(ctn, num_epoch):
    torch.cuda.empty_cache()

    #############################
    #  Train Code_Discriminator #
    #############################
    for p in D.parameters():
        p.requires_grad = False
    for p in CD.parameters():
        p.requires_grad = True
    for p in E.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False

    for iter in range(cd_iter):
        cd_optimizer.zero_grad()

        real_imgs = gen_img.__next__()
        num_img = real_imgs.size(0)
        with torch.no_grad():
            real_imgs = Variable(real_imgs).cuda()
            z_rand = Variable(torch.randn(num_img, z_dimension)).cuda()

        z_hat = E(real_imgs).view(num_img, -1)

        gradient_penalty_cd = calc_gradient_penalty(
            CD, z_hat.data, z_rand.data, w=weight_of_penalty)

        CD_loss = -CD(z_rand).mean() + CD(z_hat).mean()+gradient_penalty_cd
        CD_loss.backward(retain_graph=True)
        cd_optimizer.step()

    #############################
    #        Train Encoder      #
    #############################
    for p in D.parameters():
        p.requires_grad = False
    for p in CD.parameters():
        p.requires_grad = False
    for p in E.parameters():
        p.requires_grad = True
    for p in G.parameters():
        p.requires_grad = False

    for iter in range(e_iter):
        e_optimizer.zero_grad()

        real_imgs = gen_img.__next__()
        num_img = real_imgs.size(0)
        with torch.no_grad():
            real_imgs = Variable(real_imgs).cuda()

        z_hat = E(real_imgs).view(num_img, -1)

        e_loss = -CD(z_hat).mean()
        e_loss.backward(retain_graph=True)

        e_optimizer.step()

    #############################
    #    Train Discriminator    #
    #############################
    for p in D.parameters():
        p.requires_grad = True
    for p in CD.parameters():
        p.requires_grad = False
    for p in E.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False

    for iter in range(d_iter):
        d_optimizer.zero_grad()
        real_imgs = gen_img.__next__()
        num_img = real_imgs.size(0)
        with torch.no_grad():
            real_imgs = Variable(real_imgs).cuda()
            z_rand = Variable(torch.randn(num_img, z_dimension)).cuda()
        z_hat = E(real_imgs).view(num_img, -1)
        x_hat = G(z_hat)
        x_rand = G(z_rand)

        gradient_penalty_rand = calc_gradient_penalty(
            D, real_imgs.data, x_rand.data, w=weight_of_penalty)
        gradient_penalty_hat = calc_gradient_penalty(
            D, real_imgs.data, x_hat.data, w=weight_of_penalty)

        d_loss = -2 * D(real_imgs).mean() + D(x_hat).mean() + D(x_rand).mean()
        d_loss = d_loss + gradient_penalty_rand + gradient_penalty_hat
        d_loss.backward(retain_graph=True)

        d_optimizer.step()

    #############################
    #      Train Generator      #
    #############################
    for p in D.parameters():
        p.requires_grad = False
    for p in CD.parameters():
        p.requires_grad = False
    for p in E.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = True

    for iter in range(g_iter):
        g_optimizer.zero_grad()
        real_imgs = gen_img.__next__()
        num_img = real_imgs.size(0)
        with torch.no_grad():
            real_imgs = Variable(real_imgs).cuda()
            z_rand = Variable(torch.randn(num_img, z_dimension)).cuda()
        z_hat = E(real_imgs).view(num_img, -1)
        x_hat = G(z_hat)
        x_rand = G(z_rand)

        l1_loss = criterion_l1(x_hat, real_imgs)

        g_loss = -D(x_hat).mean() - D(x_rand).mean() + weight_of_l1*l1_loss
        g_loss.backward(retain_graph=True)

        g_optimizer.step()

    #############################
    #       Visualization       #
    #############################
    if (epoch + 1) % 50 == 0:
        print('[{}/{}]'.format(epoch, num_epoch),
              'D: {:<8.3}'.format(d_loss.mean().item()),
              'Code: {:<8.3}\n'.format(CD_loss.mean().item()),
              'En: {:<8.3}'.format(e_loss.mean().item()),
              'Ge: {:<8.3}\n'.format(g_loss.mean().item()),
              )
        with open('./log/' + savePath + '.txt', 'a') as log:
            log.write('[{}/{}] D: {:<8.3}, En: {:<8.3}, Ge: {:<8.3},Code: {:<8.3}\n'
                      .format(epoch, num_epoch, d_loss.mean().item(), e_loss.mean().item(), g_loss.mean().item(), CD_loss.mean().item()))
        writer.add_scalar(
            'D_loss', d_loss.mean().item(), global_step=epoch)
        writer.add_scalar(
            'E_loss', e_loss.mean().item(), global_step=epoch)
        writer.add_scalar(
            'CD_loss', CD_loss.mean().item(), global_step=epoch)
        writer.add_scalar(
            'G_loss', g_loss.mean().item(), global_step=epoch)

    #############################
    #      Save Breakpoint      #
    #############################
    if (epoch+1) % 5000 == 0:
        torch.save(G.state_dict(), './dict/' +
                   savePath + '/' + savePath + '_'+str(epoch) + '_G.pth')
        torch.save(D.state_dict(), './dict/' +
                   savePath + '/' + savePath + '_D.pth')
        torch.save(E.state_dict(), './dict/' +
                   savePath + '/' + savePath + '_E.pth')
        torch.save(CD.state_dict(), './dict/' +
                   savePath + '/' + savePath + '_CD.pth')

    #############################
    #         Image Save        #
    #############################
    if (epoch + 1) % 1000 == 0:
        real_img = to_img(x_hat.cpu())
        save_image(real_img, './saved_images/' +
                   savePath + '/recon_image_%d.png' % epoch)
        del real_img

        fake_img = to_img(x_rand.cpu())
        save_image(fake_img, './saved_images/' +
                   savePath+'/fake_image_%d.png' % epoch)
        del fake_img
