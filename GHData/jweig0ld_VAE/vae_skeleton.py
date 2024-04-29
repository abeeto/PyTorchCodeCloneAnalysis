import torch
import math
import time
import tensorboard_logger as tl
from torch import nn
from datetime import date
from typing import List
from torch.utils.tensorboard import SummaryWriter

""" 
Inspiration from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py 
"""


def validate_layer(module: List, three_dimensional: bool):
    if three_dimensional:
        assert len(module) == 8, "Number of elements in 3D layer is invalid"
    else:
        assert len(module) == 6, "Number of elements in 2D layers invalid"


def clean_layers(layers: List, three_dimensional: bool):
    """
    :param three_dimensional: bool indicating whether or not the layer format is for a 3d network
    :type layers: List
    :return layers: List with all instances of None converted to tuples of 0s.
    """
    if three_dimensional:
        for module in layers:
            validate_layer(module, three_dimensional)
            if module[6] is None:
                module[6] = (0, 0, 0)
            if module[7] is None:
                module[7] = (0, 0, 0)
        return layers

    for module in layers:
        validate_layer(module, three_dimensional)
        if module[4] is None:
            module[4] = (0, 0)
        if module[5] is None:
            module[5] = (0, 0)
    return layers


def get_result_dim(img_dim: int, kernel_size: int, stride: int, padding: int):
    """ Assumes that the image in question is square """
    assert kernel_size > 0, "kernel_size is less than or equal to 0"
    assert img_dim > 0, "img_dimension is less than or equal to 0"
    assert stride > 0, "stride is less than or equal to 0"
    return math.floor(((img_dim - kernel_size + 2 * padding) / stride) + 1)


def get_filepaths(json_data):
    today = date.today()
    today = today.strftime("%b-%d-%Y")
    model_filepath = json_data['model_filepath']
    log_filepath = json_data['log_filepath']
    three_d = json_data['3D']
    epochs = json_data['epochs']
    batch_size = json_data['batch_size']
    gamma_vae = json_data['gamma_vae']
    gamma = json_data['gamma']
    beta_vae = json_data['beta_vae']
    beta = json_data['beta']
    learning_rate = json_data['learning_rate']
    latent_dim = json_data['latent_dim']
    c_start = json_data['c_start']
    c_finish = json_data['c_finish']

    three_d = "3d" if three_d else "2d"
    if beta_vae:
        vae_type = "beta-{}".format(beta)
    elif gamma_vae:
        vae_type = "gamma-{}-{}-{}".format(gamma, c_start, c_finish)
    else:
        vae_type = "standard"

    model_config = '{}_{}_{}_e-{}_bs-{}_ld-{}_lr-{}'.format(today, three_d, vae_type, epochs, batch_size, latent_dim,
                                                            learning_rate)

    model_filepath = model_filepath + model_config
    log_filepath = log_filepath + model_config
    return model_filepath, log_filepath


def train(model, optimizer, train_loader, test_loader, control_batch, test_batch):
    step = 0
    model = model.float()
    training_set_size, testing_set_size = model.json_data['training_set_size'], model.json_data['testing_set_size']
    images = control_batch
    test_images = test_batch

    for epoch in range(1, model.epochs + 1):

        model.train()
        train_loss = 0
        start_time = time.time()

        # Training set
        for data in train_loader:
            # Try to make sure that training_set_size % batch_size == 0 to avoid bugs here
            batch_size = data.shape[0]
            step += 1
            optimizer.zero_grad()
            if step % 1000 == 0 or step == 1:
                start_time = time.time()
            recon_batch, probs, mu, logvar = model(data)
            loss, BCE, KLD = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            if step % 1000 == 0 or step == 1:
                end_time = time.time()
                # Log various statistics in here
                tl.log_grads(model, step)
                tl.log_images_3d(model, images, test_batch, step)
                tl.log_performance(model.writer, batch_size, start_time, end_time, step)
                print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / batch_size))

        # Test batch
        model.eval()
        test_loss = 0
        data_list = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch_size = data.shape[0]
                recon_batch, probs, mu, logvar = model(data)
                data_list.append(recon_batch)
                t_loss, t_BCE, t_KLD = model.loss_function(recon_batch, data, mu, logvar)
                test_loss += t_loss.item()

            if step % 1000 == 0:
                print('====> Test set loss: {:.4f}'.format(test_loss / batch_size))

        # Log the train and test loss
        z = 1 if not model.three_d else model.z_dim

        # TODO: Log C value in the event that the model is a gamma VAE

        train_loss = train_loss / (training_set_size * z * model.x_dim ** 2)
        test_loss = test_loss / (testing_set_size * z * model.x_dim ** 2)
        model.writer.add_scalar('Train Loss per Pixel', train_loss, step)
        model.writer.add_scalar('Test Loss per Pixel', test_loss, step)


class Conv2DVAE(nn.Module):
    def __init__(self, x_dim: int, layers: List, in_channels: int, latent_dim: int, device: torch.device, json_data,
                 **kwargs):
        """
        :param x_dim: Assumes images are square, x_dim is the length of a side of the square.
        :param layers: List<List> of length n, where n defines the number of layers in the encoder and decoder.
                       Each list in this list must follow the following order:
                       [in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:tuple,
                       output_padding:tuple]
        :param in_channels: Number of input channels per image
        :param latent_dim: Size of the latent dimension (mu and logvar vectors) / encoded representation.
        :param device: torch.device object defining the device you want to train on.
        :param json_data: The json.load result of the model's JSON config file. Any values in there are accessible
                          through here with dictionary syntax.
        :param kwargs:
        """
        super(Conv2DVAE, self).__init__()

        new_layers = clean_layers(layers, False)
        model_filepath, log_filepath = get_filepaths(self)

        self.x_dim = x_dim
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.layers = new_layers
        self.device = device
        self.json_data = json_data
        self.three_d = False
        self.model_filepath = model_filepath
        self.log_filepath = log_filepath
        self.writer = SummaryWriter(self.log_filepath)

        # Encoder
        modules = []
        current_x_dim = x_dim
        current_channels = in_channels
        for module in self.layers:
            modules.append(
                nn.Conv2d(in_channels=module[0],
                          out_channels=module[1],
                          kernel_size=module[2],
                          stride=module[3],
                          padding=module[4]),
            )
            modules.append(
                nn.ReLU()
            )
            current_x_dim = get_result_dim(img_dim=current_x_dim, kernel_size=module[2], stride=module[3],
                                           padding=module[4][0])
            current_channels = module[1]

        self.encoder = nn.Sequential(*modules)

        self.mu_layer = nn.Conv2d(in_channels=current_channels, out_channels=latent_dim, kernel_size=current_x_dim,
                                  stride=1)
        self.logvar_layer = nn.Conv2d(in_channels=current_channels, out_channels=latent_dim, kernel_size=current_x_dim,
                                      stride=1)
        self.transition_layer = nn.ConvTranspose2d(in_channels=latent_dim, out_channels=current_channels,
                                                   kernel_size=current_x_dim, stride=1)

        # Decoder
        modules = []
        layers.reverse()

        for i in range(len(self.layers) - 1):
            module = self.layers[i]
            modules.append(
                nn.ConvTranspose2d(in_channels=module[1],
                                   out_channels=module[0],
                                   kernel_size=module[2],
                                   stride=module[3],
                                   output_padding=module[5]),
            )
            modules.append(
                nn.ReLU()
            )

        # Final Layer
        module = layers[-1]
        modules.append(
            nn.ConvTranspose2d(in_channels=module[1],
                               out_channels=module[0],
                               kernel_size=module[2],
                               stride=module[3],
                               output_padding=module[5],
                               )
        )
        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state):
        post_encoder = self.encoder(state)
        mu, logvar = self.mu_layer(post_encoder), self.logvar_layer(post_encoder)
        z = self.reparameterize(mu, logvar)
        z = self.transition_layer(z)
        y = self.decoder(z)

        if y.shape[1:] != (self.in_channels, self.x_dim, self.x_dim):
            y = y[:, :self.in_channels, :self.x_dim, :self.x_dim]

        probs = torch.sigmoid(y)
        return y, probs, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        """
        :param recon_x: Reconstructed batch of shape (N, C, H, W)
        :param x: Original input batch of shape (N, C, H, W)
        :param mu: Encoded mu representation of x.
        :param logvar: Encoded logvar representation of x.
        :param kwargs: Expecting beta=(scalar), gamma=(scalar), C=(scalar).
        :return: BCE (Binary Cross-Entropy) Loss (scalar). KLD (Kullback-Leibler Divergence) loss (scalar).
                 result: Linear combination of BCE and KLD based on whether the network is gamma, beta or
                 standard VAE.
        """
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        BCE = criterion(recon_x, x)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if kwargs.get('beta') is not None:
            beta = kwargs.get('beta')
            KLD *= beta
            result = BCE + KLD
            return result, BCE, KLD

        elif kwargs.get('gamma') is not None:
            gamma = kwargs.get('gamma')
            C = kwargs.get('C')
            result = (BCE + gamma * torch.abs(KLD - C))
            return result, BCE, KLD

        result = BCE + KLD
        return result, BCE, KLD


class Conv3dVAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, channels: int, layers: List, latent_dim: int, device: torch.device,
                 json_data,
                 **kwargs):
        """
        :param x_dim: Assuming input images are square, this defines the length of a side of the square.
        :param z_dim: The number of images in a block of input, the z dimension of the cuboid.
        :param channels: The number of input channels in each image.
        :param layers: List of lists of length n, where n is the number of layers you wish to implement in encoder and decoder.
                       Each tuple has the following structure:
                        (in_channels, out_channels, kernel_size_x, kernel_size_z, stride_x, stride_z, padding:tuple, output_padding:tuple).
                        
                        Note: The padding parameter should only be used for the forward pass - use the output_padding in the
                        backward pass. If you need to add padding to the beginning of a layer during the backward pass, 
                        change the value of the previous layer's output_padding property instead of using the padding
                        parameter.
        :param latent_dim: Integer describing the number of latent dimensions (size of the mu and logvar encoded representations)
        :param device: Torch.device object describing the device to train the model on.
        :param json_data: The json.load result of the model's JSON config file. Any values in there are accessible
                          through here with dictionary syntax.
        :param kwargs:
        """
        super(Conv3dVAE, self).__init__()
        new_layers = clean_layers(layers, True)

        self.device = device
        self.latent_dim = latent_dim
        self.layers = new_layers
        self.channels = channels
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.json_data = json_data
        self.three_d = True
        model_filepath, log_filepath = get_filepaths(self.json_data)
        self.model_filepath = model_filepath
        self.log_filepath = log_filepath
        self.writer = SummaryWriter(self.log_filepath)

        modules = []
        current_channels = channels
        current_x_dim = x_dim
        current_z_dim = z_dim

        # Encoder
        for module in self.layers:
            modules.append(
                nn.Conv3d(in_channels=module[0],
                          out_channels=module[1],
                          kernel_size=(module[3], module[2], module[2]),
                          stride=(module[5], module[4], module[4]),
                          padding=module[6])
            )
            modules.append(
                nn.ReLU()
            )
            current_channels = module[1]

            if module[6] is None:
                module[6] = [0, 0, 0]

            current_x_dim = get_result_dim(current_x_dim, kernel_size=module[2], stride=module[4], padding=module[6][1])
            current_z_dim = get_result_dim(current_z_dim, kernel_size=module[3], stride=module[5], padding=module[6][0])

        self.encoder = nn.Sequential(*modules)

        # Latent dimensions
        self.mu_layer = nn.Conv3d(in_channels=current_channels, out_channels=latent_dim,
                                  kernel_size=(current_z_dim, current_x_dim, current_x_dim), stride=(1, 1, 1))
        self.logvar_layer = nn.Conv3d(in_channels=current_channels, out_channels=latent_dim,
                                      kernel_size=(current_z_dim, current_x_dim, current_x_dim), stride=(1, 1, 1))
        self.transition_layer = nn.ConvTranspose3d(in_channels=latent_dim, out_channels=current_channels,
                                                   kernel_size=(current_z_dim, current_x_dim, current_x_dim),
                                                   stride=(1, 1, 1))

        layers.reverse()
        modules = []

        # Decoder
        for i in range(len(self.layers) - 1):
            module = self.layers[i]
            modules.append(
                nn.ConvTranspose3d(in_channels=module[1],
                                   out_channels=module[0],
                                   kernel_size=(module[3], module[2], module[2]),
                                   stride=(module[5], module[4], module[4]),
                                   output_padding=module[7]
                                   ),
            )
            modules.append(
                nn.ReLU()
            )

        # Final layer - no ReLU activation
        module = layers[-1]
        modules.append(
            nn.ConvTranspose3d(
                in_channels=module[1],
                out_channels=module[0],
                kernel_size=(module[3], module[2], module[2]),
                stride=(module[5], module[4], module[4]),
                padding=(0, 0, 0),
                output_padding=module[7]
            )
        )

        self.decoder = nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)
        z = self.transition_layer(z)
        y = self.decoder(z)

        if y.shape[1:] != (self.channels, self.x_dim, self.z_dim):
            y = y[:, :self.channels, :self.z_dim, :self.x_dim, :self.x_dim]

        probs = torch.sigmoid(y)
        return y, probs, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        """
        :param recon_x: Reconstructed batch of shape (N, C, Z, H, W)
        :param x: Original input batch of shape (N, C, Z, H, W)
        :param mu: Encoded mu representation of x.
        :param logvar: Encoded logvar representation of x.
        :param kwargs: Expecting beta=(scalar), gamma=(scalar), C=(scalar).
        :return: BCE (Binary Cross-Entropy) Loss (scalar). KLD (Kullback-Leibler Divergence) loss (scalar).
                 result: Linear combination of BCE and KLD based on whether the network is gamma, beta or
                 standard VAE.
        """
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        BCE = criterion(recon_x, x)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if kwargs.get('beta') is not None:
            beta = kwargs.get('beta')
            KLD *= beta
            result = BCE + KLD
            return result, BCE, KLD

        elif kwargs.get('gamma') is not None:
            gamma = kwargs.get('gamma')
            C = kwargs.get('C')
            result = (BCE + gamma * torch.abs(KLD - C))
            return result, BCE, KLD

        result = BCE + KLD
        return result, BCE, KLD
