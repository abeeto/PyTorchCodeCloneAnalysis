""" #CycleGAN, Image-to-Image Translation

"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride = 2,
    padding = 1,
    batch_norm = True
    ):
    """ Create a convolutional layer with an optional batch normalization layer.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """ Discriminator

        Define a Discriminator for Cycle GAN.
    """
    def __init__(
        self, 
        input_in_channels = 3, 
        conv_dim = 64
        ):
        """ Initialize Discriminator.
            Arguments:
            input_in_inchannels : int
                Number of channels for the input
            conv_dim : int
                Number of output channels for the first conv-layer.
        """
        super(Discriminator, self).__init__()

        self.input_in_channels = input_in_channels
        self.conv_dim = conv_dim

        # Define all convolutional layers
        self.conv1 = conv(self.input_in_channels, self.conv_dim*1, 4, 2, 1, batch_norm = False)
        self.conv2 = conv(self.conv_dim*1, self.conv_dim*2, 4, 2, 1, batch_norm = True)
        self.conv3 = conv(self.conv_dim*2, self.conv_dim*4, 4, 2, 1, batch_norm = True)
        self.conv4 = conv(self.conv_dim*4, self.conv_dim*8, 4, 2, 1, batch_norm = True)

        # kernel_size = 128/2/2/2/2 = 8
        self.conv5 = conv(self.conv_dim*8, self.conv_dim*16, 4, 2, 1, batch_norm = False)

        self.fc_last = nn.Linear(self.conv_dim*16*4*4, 1)

        self.relu = nn.ReLU()

    def forward(
        self, 
        x
        ):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x.view(-1, self.conv_dim*16*4*4)
        out = self.fc_last(x)
        return out


class ResidualBlock(nn.Module):
    """ ResidualBlock

        Compute a skip connection.
    """

    def __init__(
        self,
        conv_dim
        ):
        super(ResidualBlock, self).__init__()
        self.conv_dim = conv_dim
        
        self.conv1 = conv(self.conv_dim, self.conv_dim, 3, 1, 1, batch_norm= True)
        self.conv2 = conv(self.conv_dim, self.conv_dim, 3, 1, 1, batch_norm= True)
        self.relu = nn.ReLU()

    def forward(
        self,
        x
        ):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)

        return y + x


def deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    batch_norm = True
    ):
    """ Create a transpose-convolutional layer with an optional batch normalization layer.
    """
    layers = []
    deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(deconv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class CycleGenerator(nn.Module):
    """ Generator for Cycle GAN
    """
    def __init__(
        self,
        input_in_channels = 3,
        conv_dim = 64,
        n_res_blocks = 6
        ):
        """ Initialize Cycle GAN Generator
            Arguments
            ---------
            conv_dim : int
                Size of output channel for the first convolutional layer.
            n_res_blocks : int
                Number of residual blocks in the generator
        """
        super(CycleGenerator, self).__init__()
        self.input_in_channels = input_in_channels
        self.conv_dim = conv_dim
        self.n_res_blocks = n_res_blocks

        # Define the encoder part of the generator
        self.encoder = nn.Sequential(
            conv(self.input_in_channels, self.conv_dim, 4, 2, 1, batch_norm = True),
            nn.ReLU(),
            conv(self.conv_dim, self.conv_dim*2, 4, 2, 1, batch_norm = True),
            nn.ReLU(),
            conv(self.conv_dim*2, self.conv_dim*4, 4, 2, 1, batch_norm = True),
            nn.ReLU()                        
            )

        # Define the resnet part of the generator
        layers = [ResidualBlock(self.conv_dim*4) for _ in range(self.n_res_blocks)]
        self.resnet = nn.Sequential(
            *layers
            )

        # Define the decoder part of the generator
        self.decoder = nn.Sequential(
            deconv(self.conv_dim*4, self.conv_dim*2, 4, 2, 1, batch_norm = True),
            nn.ReLU(),
            deconv(self.conv_dim*2, self.conv_dim*1, 4, 2, 1, batch_norm = True),
            nn.ReLU(),
            deconv(self.conv_dim*1, 3, 4, 2, 1, batch_norm = False),
            nn.Tanh()
        )


    def forward(
        self, 
        x):
        x = self.encoder(x)
        x = self.resnet(x)
        x = self.decoder(x)
        return x


# Losses for Cycle GAN
def real_mse_loss(
    D_out
    ):
    """ Calculate loss for a real image 
         Arguments
         ---------
         D_out : a float
            Output of Discriminator for a fake image

         Returns
         -------
            loss = (D_out - 1) ** 2
    """
    return torch.mean((D_out - 1.)**2)


def fake_mse_loss(
    D_out
    ):
    """ Calculate loss for a fake image 
        Arguments
        ---------
        D_out : a float
            Output of Discriminator for a fake image

        Returns
        -------
            loss = (D_out - 0) ** 2
    """
    return torch.mean(D_out**2)

def cycle_consistency_loss(
    real_im,
    reconstructed_im,
    lambda_weight
    ):
    """ Calculate cycle consistency loss
        Arguments
        ---------
        real_im : array_like = image
            Real input image for the generator
        reconstructed_im : array_like = image
            Image reconstructed throughout generators
        lambda_weight : float
            A weight-factor of cycle consistency loss

        Returns
        -------
        cycle consistency loss
    """
    return lambda_weight*torch.mean(torch.abs(real_im - reconstructed_im))





# Helper Functions

def get_data_loader(
    image_type,
    image_dir,
    image_size = 128,
    batch_size = 16,
    num_workers = 0
    ):
    """Returns training and test data loaders for a given image type. 
       These images will be resized to 128x128x3, by default, 
       converted into Tensors, and normalized.
    """
    # resize and normalize the images
    transform = transforms.Compose([transforms.Resize(image_size), # resize to 128x128
                                    transforms.ToTensor()#,
                                    # transforms.Normalize(
                                    #     mean = [0.485, 0.456, 0.406],
                                    #     std=[0.229, 0.224, 0.225]
                                    # )
                                    ])

    # get training and test directories
    image_path = os.path.join('.', image_dir)
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# helper scale function
def scale(
    x, 
    feature_range=(-1, 1)
    ):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-255.'''
    
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x    


def create_model(
    input_in_channels = 3,
    generator_conv_dim = 64,
    discriminator_conv_dim = 64,
    n_res_blocks = 6
    ):
    """ Instantiate two generators and two discriminator for Cycle GAN
    
        Arguments:
        input_in_channels : int
            Number of channels in input
        generator_conv_dim : int
            Size of output channels of the first conv-layer of the Generator
        discriminator_conv_dim : int
            Size of output channels of the first conv-layer of the Discriminator

        Returns
        -------
        G_XtoY, G_YtoX, D_X, D_Y
    """
    # Instantiate Generators
    G_XtoY = CycleGenerator(input_in_channels, generator_conv_dim, n_res_blocks)
    G_YtoX = CycleGenerator(input_in_channels, generator_conv_dim, n_res_blocks)

    # Instantiate Discriminators
    D_X = Discriminator(input_in_channels, discriminator_conv_dim)
    D_Y = Discriminator(input_in_channels, discriminator_conv_dim)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y


# helper function for printing the model architecture
def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                     G_XtoY                    ")
    print("-----------------------------------------------")
    print(G_XtoY)
    print()

    print("                     G_YtoX                    ")
    print("-----------------------------------------------")
    print(G_YtoX)
    print()

    print("                      D_X                      ")
    print("-----------------------------------------------")
    print(D_X)
    print()

    print("                      D_Y                      ")
    print("-----------------------------------------------")
    print(D_Y)
    print()