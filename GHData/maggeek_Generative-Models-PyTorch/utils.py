from torchvision.utils import save_image
from torch.autograd import Variable
from config import args
from random import randint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def load_network(net_name, net):
    network = net
    try:
        network.load_state_dict(torch.load('{}/{}.pth'.format(args.folder_name, net_name)))
    except:
        pass
    if cuda:
        network.cuda()
    network.apply(weights_init)
    return network


def load_images(path):
    # load images and scale in range(-1,1)
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = datasets.ImageFolder(root=path, transform=transform)
    print(len(images))
    images_loader = torch.utils.data.DataLoader(images, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return images_loader


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), args.latent_dim))))
    z = sampled_z * std + mu
    return z


def save_loss(epoch, batch, how_many_batches, loss, name):
    file = open('{}/loss_{}.txt'.format(args.folder_name, name), 'a')
    file.write("[Epoch %d/%d] [Batch %d/%d] [Rec loss: %f]" % (epoch, args.n_epochs, batch, how_many_batches, loss) + "\n")
    file.close()


def save_plot(epoch, loss, name):
    """
    Plot loss over epochs
    :param epoch: current epoch
    :param loss: list of losses over epochs
    :param name: plot file name
    """
    x = list(range(0, epoch+1))
    plt.plot(x, loss, '-b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('{} Loss'.format(name))
    plt.savefig('{}/loss_{}.png'.format(args.folder_name, name), bbox_inches='tight')
    plt.close()


def save_images(imgs, name, n_row, batches_done):
    """
    Save a grid of images
    :param imgs: images to save
    :param name: image file name
    :param n_row: how many images
    :param batches_done: number of batches of training so far
    """
    save_image(imgs, '{}/{}_{}.png'.format(args.folder_name, batches_done, name), nrow=n_row, normalize=True)


def gaussian(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise


# custom weights initialization called on generator and disciminator
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
    #if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
    #elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def print_parameters(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    print('{:,} total parameters in {}.'.format(total_params, name))
    #print(f'{total_params:,} total parameters.')
