# VAE Tutorial from https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=_o1KPYut49DJ

# Imports
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


# Parameter Settings
# 2-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
latent_dims = 16
Epochs = 40
train_batch_size = 128
test_batch_size = 128
capacity = 64
learning_rate = 1e-3
variational_beta = 1
use_gpu = True

# # 10-d latent space, for comparison with non-variational auto-encoder
# latent_dims = 10
# Epochs = 10
# batch_size = 128
# test_batch_size = 128
# capacity = 64
# learning_rate = 1e-3
# variational_beta = 1
# use_gpu = True

# Path for Data
data_path = './data'
# Path to save and load model
if latent_dims == 2:
    net_path = './models/VAE_net.pth'
else:
    net_path = './models/VAE_net' + str(latent_dims) + '.pth'

# set up the divice (GPU or CPU) via input prompt
cuda_true = input("Use GPU? (y) or (n)?")
if cuda_true == "y":
    device = "cuda"
else:
    device = "cpu"
print("Device:", device)


# save a networks parameters for future use
def save_network(net: nn.Module, path):
    # save the network?
    save = input("Save net? (y) or (n)?")
    if save == "y":
        torch.save(net.state_dict(), path)
    else:
        pass


# load an existing network's parameters and safe them into the just created net
def load_network(net: nn.Module, path):
    # save the network?
    load = input("Load Network? (y) or (n)?")
    if load == "y":
        net.load_state_dict(torch.load(net_path))
    else:
        pass


# Dataloading MNIST

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
testset = torchvision.datasets.MNIST(root=data_path, train=False, download=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=0)


# VAE Definition
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)   # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)           # flatten batch of multi channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity * 2, 7, 7)       # unflatten batch of feature vectors to a batch of multi channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid((self.conv1(x)))              # last layer before output is sigmoid, since we are using BCE as recon loss
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparametrization trick!
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction="sum")

    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence


vae = VAE()
try:
    load_network(vae, net_path)
except:
    print("no network saved yet under path: %s" % str(net_path))
vae.to(device=device)

num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print("Number of parameters: %d" % num_params)

# Training
optimizer = optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

train_loss_avg = []


def train(num_epochs):
    # set to training mode
    vae.train()
    print("Training ...")

    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, _ in trainloader:
            image_batch = image_batch.to(device=device)

            # vae reconsturction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

            # reconstruction error
            loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print("Epoch [%d / %d] average reconstruction loss: %f" % (epoch + 1, num_epochs, train_loss_avg[-1]))


# Evaluation
def evaluate_vae(dataloader: DataLoader):
    vae.eval()
    if dataloader.dataset.train:
        set_name = "Trainloader"
    else:
        set_name = "Testloader"

    test_loss_avg, num_batches = 0, 0
    for image_batch, _ in dataloader:

        with torch.no_grad():
            image_batch = image_batch.to(device=device)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

            # reconstruction error
            loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            test_loss_avg += loss.item()
            num_batches += 1

    test_loss_avg /= num_batches
    print("Average reconstruction error for %s: %f" % (set_name, test_loss_avg))


# execute training!
train_true = input("Train network? (y) or (n)?")
if train_true == "y":
    train(Epochs)

    # plot training curve
    def plot_train_curve():
        # plt.ion()

        fig = plt.figure()
        plt.plot(train_loss_avg)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    plot_train_curve()

    evaluate_vae(trainloader)
    evaluate_vae(testloader)

    # save the network?
    save = input("Save net? (y) or (n)?")
    if save == "y":
        torch.save(vae.state_dict(), net_path)
    else:
        pass

else:
    pass


# image helpers
def to_img(x):
    x = x.clamp(0,1)
    return x


def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def visualize_reconstruction(dataloader: DataLoader):
    # plt.ion()
    vae.eval()

    # This function takes as an input the images to reconstruct
    # and the name of the model with which the reconstructions
    # are performed
    def visualise_output(images, model):
        f1 = plt.figure(1)
        with torch.no_grad():
            images = images.to(device=device)
            images, _, _ = model(images)
            images = images.cpu()
            images = to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
            f1 = plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            f1 = plt.title("VAE reconstruction")
            # plt.show()
            return f1

    images, labels = iter(dataloader).next()

    # First visualise the original image
    f2 = plt.figure(2)
    show_image(torchvision.utils.make_grid(images[1:50], 10, 5))
    plt.title("Original images")
    # plt.show()

    # Reconstruct and visualize the images using vae
    visualise_output(images, vae)

    plt.show()


def interpolate_latent_space(dataloader: DataLoader):
    vae.eval()

    def interpolation(lambda1, model: VAE, img1, img2):

        with torch.no_grad():
            # latent vector of the first image
            img1 = img1.to(device=device)
            latent_1, _ = model.encoder(img1)

            # latent vector of the second image
            img2 = img2.to(device=device)
            latent_2, _ = model.encoder(img2)

            # interpolation of the two latent vectors / between 2 images from the latent space
            inter_latent = lambda1 * latent_1 + (1 - lambda1) * latent_2

            # reconstruct interpolated image
            inter_image = model.decoder(inter_latent)
            inter_image = inter_image.cpu()
            return inter_image

    # sort part of the test set by digit
    digits = [[] for _ in range(10)]
    for img_batch, label_batch in dataloader:
        for i in range(img_batch.size(0)):
            digits[label_batch[i]].append(img_batch[i:i+1])
        if sum(len(d) for d in digits) >= 1000:
            break

    # interpolation lambdas
    lambda_range = np.linspace(0, 1, num=20)

    fig, axs = plt.subplots(4, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace = 0.5, wspace = 0.001)
    axs = axs.ravel()

    for ind, l in enumerate(lambda_range):
        inter_image = interpolation(float(l), vae, digits[0][0], digits[7][0])

        inter_image = to_img(inter_image)

        image = inter_image.numpy()

        axs[ind].imshow(image[0, 0, :, :], cmap="gray")
        axs[ind].set_title("lambda_val=" + str(round(l, 2)))
    plt.show()


def sample_latent_vector():
    vae.eval()
    with torch.no_grad():
        # sample latent vectors from the normal distribution
        latent = torch.randn(128, latent_dims, device=device)

        # reconstruct images from the latent vectors
        img_recon = vae.decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(5, 5))
        show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
        plt.show()


# ignore for now more than 2D!
def show_2D_latent_space():
    if latent_dims != 2:
        print("Latent space is bigger than 2!")

    with torch.no_grad():
        # create a sample grid in 2d latent space
        latent_x = np.linspace(-1.5, 1.5, 20)
        latent_y = np.linspace(-1.5, 1.5, 20)
        latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
        for i, lx in enumerate(latent_x):
            for j, ly in enumerate(latent_y):
                latents[j, i, 0] = lx
                latents[j, i, 1] = ly
        latents = latents.view(-1, 2)       # flatten grid into a batch

        # reconstruct images from latent vectors
        latents = latents.to(device=device)
        img_recon = vae.decoder(latents)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(10, 10))
        show_image(torchvision.utils.make_grid(img_recon.data[:400], 20, 5))
        plt.show()


# works
# visualize_reconstruction(trainloader)

# works
# interpolate_latent_space(testloader)

sample_latent_vector()
