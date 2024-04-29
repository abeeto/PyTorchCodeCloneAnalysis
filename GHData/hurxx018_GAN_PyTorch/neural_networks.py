import pickle as pkl

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Discriminator(nn.Module):
    
    def __init__(
        self,
        input_size,
        hidden_dim,
        output_size
        ):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(
        self, 
        x
        ):
        # flatten image
        x = x.view(-1, self.input_size)
        # pass x through all layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)

        return x


class Generator(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_dim,
        output_size
        ):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def forward(
        self, 
        x
        ):
        # pass x through all layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        # final layer should have tanh applied
        x = self.tanh(x)

        return x


def real_loss(
    D_out, 
    smooth = False,
    train_on_gpu = False
    ):
    """ Loss for real data
        Arguments
        ---------
        D_out : outputs of Discriminator for real data
        smooth : label-smoothing

        Returns
        -------
        loss
    """
    batch_size = D_out.shape[0]
    if smooth:
        p = 0.9
    else:
        p = 1.0

    criterion = nn.BCEWithLogitsLoss()
    targets = torch.ones(batch_size)*p
    if train_on_gpu:
        targets = targets.cuda()
    loss = criterion(D_out.squeeze(), targets)

    return loss

def fake_loss(
    D_out,
    train_on_gpu = False
    ):
    """ Loss for fake data from Generator
        Arguments
        ---------
        D_out : outputs of Discriminator for fake data

        Returns
        -------
        loss
    """
    batch_size = D_out.shape[0]

    criterion = nn.BCEWithLogitsLoss()
    targets = torch.zeros(batch_size)
    if train_on_gpu:
        targets = targets.cuda()

    loss = criterion(D_out.squeeze(), targets)

    return loss


def train(
    train_loader,
    D,
    G,
    n_epochs = 10,
    lr = 0.001,
    print_every = 400,
    sample_size = 16,
    random_seed = None,
    train_on_gpu = None
    ):
    if train_on_gpu:
        D, G = D.cuda(), G.cuda()

    d_optimizer = optim.Adam(D.parameters(), lr = lr)
    g_optimizer = optim.Adam(G.parameters(), lr = lr)

    samples  = []
    d_losses = []
    g_losses = []

    rng = np.random.default_rng(random_seed)

    fixed_z = rng.uniform(-1, 1, size=(sample_size, G.input_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    D.train()
    G.train()
    for epoch in range(n_epochs):

        for i_batch, (real_images, _) in enumerate(train_loader):
            batch_size = len(real_images)

            if train_on_gpu:
                real_images = real_images.cuda()

            # Important rescaling step
            real_images = real_images*2 - 1

            # Train the Discriminator

            outputs = D(real_images)
            if train_on_gpu:
                outputs = outputs.cuda()
            r_loss = real_loss(outputs, True, train_on_gpu)


            z = rng.uniform(-1, 1, (batch_size, G.input_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            outputs = D(fake_images)

            # compute the discriminator losses on fake images
            f_loss = fake_loss(outputs, train_on_gpu)

            d_loss = r_loss + f_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train the Generator

            z = rng.uniform(-1, 1, (batch_size, G.input_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = real_loss(outputs, False, train_on_gpu)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if i_batch % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        # AFTER each epoch
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        # generate and save sampled fake images
        G.eval()
        with torch.no_grad():
            samples_z = G(fixed_z)
            if train_on_gpu:
                samples_z = samples_z.cpu()
            samples.append(samples_z)
        G.train()

    # Save training losses
    np.save("g_losses", np.asarray(g_losses))
    np.save("d_losses", np.asarray(d_losses))

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return D, G
