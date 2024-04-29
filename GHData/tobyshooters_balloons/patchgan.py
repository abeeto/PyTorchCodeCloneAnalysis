import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Defines a PatchGAN discriminator
    Ref: github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(Discriminator, self).__init__()

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, False)
        ]

        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult, nf_mult_prev = min(2 ** n, 8), nf_mult
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult, nf_mult_prev = min(2 ** n_layers, 8), nf_mult
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return torch.sigmoid(output)
