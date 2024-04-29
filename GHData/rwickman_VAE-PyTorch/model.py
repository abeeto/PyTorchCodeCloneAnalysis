import torch
import torch.nn as nn

from model_util import reparameterization_trick

class Encoder(nn.Module):
    def __init__(self, input_dim=10, z_size=256):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        
        self.act = nn.LeakyReLU()
        self.mu_out = nn.Linear(z_size, z_size)
        self.sigma_out = nn.Linear(z_size, z_size)

    
    def forward(self, x):
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.act(self.conv_4(x))
        x = torch.flatten(x, start_dim=1)
        return self.mu_out(x), self.sigma_out(x)

class Decoder(nn.Module):
    def __init__(self, input_dim=10, hidden_size=128, z_size=128):
        super().__init__()
        self.conv_tran_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_tran_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_tran_3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_tran_4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.act = nn.LeakyReLU()
        self.conv_out = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=0)
    
    def forward(self, z):
        z = z.reshape(z.shape[0], 64, 2, 2)
        x = self.act(self.conv_tran_1(z))
        x = self.act(self.conv_tran_2(x))
        x = self.act(self.conv_tran_3(x))
        x = self.act(self.conv_tran_4(x))
        x = self.conv_out(x)

        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterization_trick(mu, log_var)   
        
        return self.decoder(z), mu, log_var

    def generate(self, inp):
        x_pred = self.decoder(inp)

        return x_pred