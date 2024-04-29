import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, conv_params, in_channels):

        super(Discriminator, self).__init__()

        self.sequential = nn.Sequential()

        modules = []

        for conv_param in conv_params[:1]:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_param.filters,
                kernel_size=conv_param.kernel_size,
                stride=2,
                padding=1,
                bias=False
            )
            nn.init.kaiming_normal(conv.weight, a=0.2, nonlinearity="leaky_relu")
            modules.append(conv)
            modules.append(nn.LeakyReLU(0.2))

        for prev_conv_param, conv_param in zip(conv_params[:-1], conv_params[1:]):
            conv = nn.Conv2d(
                in_channels=prev_conv_param.filters,
                out_channels=conv_param.filters,
                kernel_size=conv_param.kernel_size,
                stride=2,
                padding=1,
                bias=False
            )
            nn.init.kaiming_normal(conv.weight, a=0.2, nonlinearity="leaky_relu")
            modules.append(conv)
            batch_norm = nn.BatchNorm2d(conv_param.filters)
            nn.init.ones_(batch_norm.weight)
            nn.init.zeros_(batch_norm.bias)
            modules.append(batch_norm)
            modules.append(nn.LeakyReLU(0.2))

        for conv_param in conv_params[-1:]:
            conv = nn.Conv2d(
                in_channels=conv_param.filters,
                out_channels=1,
                kernel_size=conv_param.kernel_size,
                stride=1,
                padding=0,
                bias=False
            )
            nn.init.xavier_normal_(conv.weight)
            modules.append(conv)
            modules.append(nn.Sigmoid())

        self.sequential = nn.Sequential(*modules)

    def forward(self, input):
        return self.sequential(input).squeeze(-1).squeeze(-1)


class Generator(nn.Module):

    def __init__(self, latent_dims, deconv_params, out_channels):

        super(Generator, self).__init__()

        modules = []

        for deconv_param in deconv_params[:1]:
            deconv = nn.ConvTranspose2d(
                in_channels=latent_dims,
                out_channels=deconv_param.filters,
                kernel_size=deconv_param.kernel_size,
                stride=1,
                padding=0,
                bias=False
            )
            nn.init.kaiming_normal(deconv.weight, a=0.0, nonlinearity="relu")
            modules.append(deconv)
            batch_norm = nn.BatchNorm2d(deconv_param.filters)
            nn.init.ones_(batch_norm.weight)
            nn.init.zeros_(batch_norm.bias)
            modules.append(batch_norm)
            modules.append(nn.ReLU())

        for prev_deconv_param, deconv_param in zip(deconv_params[:-1], deconv_params[1:]):
            deconv = nn.ConvTranspose2d(
                in_channels=prev_deconv_param.filters,
                out_channels=deconv_param.filters,
                kernel_size=deconv_param.kernel_size,
                stride=2,
                padding=1,
                bias=False
            )
            nn.init.kaiming_normal(deconv.weight, a=0.0, nonlinearity="relu")
            modules.append(deconv)
            batch_norm = nn.BatchNorm2d(deconv_param.filters)
            nn.init.ones_(batch_norm.weight)
            nn.init.zeros_(batch_norm.bias)
            modules.append(batch_norm)
            modules.append(nn.ReLU())

        for deconv_param in deconv_params[-1:]:
            deconv = nn.ConvTranspose2d(
                in_channels=deconv_param.filters,
                out_channels=out_channels,
                kernel_size=deconv_param.kernel_size,
                stride=2,
                padding=1,
                bias=False
            )
            nn.init.xavier_normal_(deconv.weight)
            modules.append(deconv)
            modules.append(nn.Tanh())

        self.sequential = nn.Sequential(*modules)

    def forward(self, input):
        return self.sequential(input.unsqueeze(-1).unsqueeze(-1))
