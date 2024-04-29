import os
import argparse
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision import utils
import model
from param import Param

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--latent_dims", type=int, default=128)
parser.add_argument("--discriminator", type=str, default="")
parser.add_argument("--generator", type=str, default="")
args = parser.parse_args()

dataset = datasets.MNIST(
    root="mnist",
    download=True,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=os.cpu_count()
)

discriminator = model.Discriminator(
    conv_params=[
        Param(filters=32, kernel_size=4),
        Param(filters=64, kernel_size=4),
        Param(filters=128, kernel_size=4),
    ],
    in_channels=1
)
print(discriminator)

if args.discriminator:
    discriminator.load_state_dict(torch.load(args.discriminator))

generator = model.Generator(
    latent_dims=args.latent_dims,
    deconv_params=[
        Param(filters=128, kernel_size=4),
        Param(filters=64, kernel_size=4),
        Param(filters=32, kernel_size=4),
    ],
    out_channels=1
)
print(generator)

if args.generator:
    generator.load_state_dict(torch.load(args.generator))

discriminator_optimizer = optim.Adam(
    params=discriminator.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)
generator_optimizer = optim.Adam(
    params=generator.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)

for epoch in range(args.num_epochs):
    for step, (reals, _) in enumerate(data_loader):

        discriminator.zero_grad()

        real_logits = discriminator(reals)
        real_loss = nn.functional.softplus(-real_logits).mean()

        latents = torch.randn(reals.size(0), args.latent_dims)
        fakes = generator(latents)
        fake_logits = discriminator(fakes.detach())
        fake_loss = nn.functional.softplus(fake_logits).mean()

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator.zero_grad()

        fake_logits = discriminator(fakes)
        fake_loss = nn.functional.softplus(-fake_logits).mean()

        generator_loss = fake_loss
        generator_loss.backward()
        generator_optimizer.step()

        print(f"[{epoch}/{args.num_epochs}]: discriminator_loss: {discriminator_loss} generator_loss: {generator_loss}")

    utils.save_image(reals, f"reals/{epoch}.png", normalize=True)
    utils.save_image(fakes.detach(), f"fakes/{epoch}.png", normalize=True)

    torch.save(discriminator.state_dict(), f"model/discriminator/epoch_{epoch}.pth")
    torch.save(generator.state_dict(), f"model/generator/epoch_{epoch}.pth")
