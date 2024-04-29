import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

#Residual down sampling block for the encoder
#Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Res_down, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_out//2, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv2d(channels_out//2, channels_out, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv2d(channels_in, channels_out, kernel_size = 3, padding=1)
        self.AvePool = nn.AvgPool2d(kernel_size = 2)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        x = F.relu(self.conv1(x))
        x = self.AvePool(x)
        x = self.conv2(x)
        x = F.relu(x + skip)
        return x

#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Res_up, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_out//2, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv2d(channels_out//2, channels_out, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv2d(channels_in, channels_out, kernel_size = 3, padding=1)

        self.UpNN = nn.Upsample(scale_factor = 2, mode = "nearest")

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        x = F.relu(self.conv1(x))
        x = self.UpNN(x)
        x = self.conv2(x)
        x = F.relu(x + skip)
        return x

class Encoder(nn.Module):
    def __init__(self, channels_in, depth = 5, z = 10):
        super(Encoder, self).__init__()
        self.down_path = nn.ModuleList([Res_down(channels_in = channels_in*(2**i), channels_out = channels_in*(2**(i+1)))\
                                        for i in range(depth)])
        self.conv_mu = nn.Conv2d(channels_in*(2**(depth)), z, 3)#2
        self.conv_logvar = nn.Conv2d(channels_in*(2**(depth)), z, 3)#2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, Train = True):
        for down_module in self.down_path:
            x = down_module(x)

        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        self.z = self.reparameterize(mu, logvar)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, channels_in, depth = 5, z = 10):
        super(Decoder, self).__init__()
        self.conv1 = Res_up(z, channels_in*(2**depth))
        self.up_path = nn.ModuleList([Res_up(channels_in = channels_in*(2**(i+1)),channels_out = channels_in*(2**i))\
                                      for i in range(depth-1, -1,-1)])
    def forward(self, x):
        x = self.conv1(x)
        for up_module in self.up_path:
          x = up_module(x)
        return x

class VAE(nn.Module):
    def __init__(self, channels_in, depth = 5, z_dim = 10):
        super(VAE, self).__init__()
        """Res VAE Network
        channels_in  = number of channels of the image
        z = the number of channels of the latent representation (for a 64x64 image this is the size of the latent vector)"""

        self.encoder = Encoder(channels_in, depth = depth, z_dim = z_dim)
        self.decoder = Decoder(channels_in, depth = depth -1, z_dim = z_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def loss_fn(recon_x, x, mu, logvar):
    # will be used later
    BCE = F.mse_loss(recon_x, x, reduction = "mean")
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

class abstract_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
