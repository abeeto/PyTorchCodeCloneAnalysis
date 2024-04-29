
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, c = 32, latent_size = 400):
    super(Generator, self).__init__()

    self.conv = nn.Sequential(
        nn.Unflatten(1, (latent_size, 1, 1)),
        nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size, kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(latent_size),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(in_channels=latent_size, out_channels=c*8, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c*8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(in_channels=c*8, out_channels=c*4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c*2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(c),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
        )

  def forward(self, latent):
    return self.conv(latent)

  def pred_image(self, latent, stats):
      pred = self.conv(latent)
      return pred * stats[1][0] + stats[0][0]

class Discriminator(nn.Module):
  def __init__(self, c = 32):
    super(Discriminator, self).__init__()
    # in: 3 x 128 x 128
    self.conv = nn.Sequential(

        nn.Conv2d(3, c, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(c, c*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c*2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(c*2, c*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c*4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(c*4, c*8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c*8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(c*8, c*8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c*8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(c*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        # out: 1 x 1 x 1

        nn.Flatten(),
        nn.Sigmoid()
        )

  def forward(self, image):
    return self.conv(image)
