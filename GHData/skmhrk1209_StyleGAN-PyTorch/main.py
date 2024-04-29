import os
import argparse
import torch
from torch import nn
from torch import optim
from torch import backends
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision import utils
from torchvision import models
import model
import metrics
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--generator_checkpoint", type=str, default="")
parser.add_argument("--discriminator_checkpoint", type=str, default="")
parser.add_argument("--mapping_network_checkpoint", type=str, default="")
parser.add_argument("--checkpoint_directory", type=str, default="checkpoints")
parser.add_argument("--event_directory", type=str, default="events")
parser.add_argument("--dataset_directory", type=str, default="cifar10")
args = parser.parse_args()

backends.cudnn.benchmark = True
device = torch.device("cuda")

latent_size = 512
mapping_network_learning_rate = 2e-5
mapping_network_beta1 = 0.0
mapping_network_beta2 = 0.99
mapping_network_epsilon = 1e-8
generator_learning_rate = 2e-3
generator_beta1 = 0.0
generator_beta2 = 0.99
generator_epsilon = 1e-8
discriminator_learning_rate = 2e-3
discriminator_beta1 = 0.0
discriminator_beta2 = 0.99
discriminator_epsilon = 1e-8
real_gradient_penalty_weight = 5.0
fake_gradient_penalty_weight = 0.0

mapping_network = model.MappingNetwork(
    embedding_param=Dict(num_embeddings=10, embedding_dim=512),
    linear_params=[
        Dict(in_features=1024, out_features=512),
        *[Dict(in_features=512, out_features=512)] * 6,
        Dict(in_features=512, out_features=512)
    ]
).to(device)
print(mapping_network)

if args.mapping_network_checkpoint:
    mapping_network.load_state_dict(torch.load(args.mapping_network_checkpoint))

generator = model.Generator(
    min_resolution=4,
    max_resolution=256,
    min_channels=16,
    max_channels=512,
    num_features=512,
    out_channels=3
).to(device)
print(generator)

if args.generator_checkpoint:
    generator.load_state_dict(torch.load(args.generator_checkpoint))

discriminator = model.Discriminator(
    min_resolution=4,
    max_resolution=256,
    min_channels=16,
    max_channels=512,
    num_classes=10,
    in_channels=3
).to(device)
print(discriminator)

if args.discriminator_checkpoint:
    discriminator.load_state_dict(torch.load(args.discriminator_checkpoint))

generator_optimizer = optim.Adam([
    dict(
        params=mapping_network.parameters(),
        lr=mapping_network_learning_rate,
        betas=(mapping_network_beta1, mapping_network_beta2),
        eps=mapping_network_epsilon
    ),
    dict(
        params=generator.parameters(),
        lr=generator_learning_rate,
        betas=(generator_beta1, generator_beta2),
        eps=generator_epsilon
    )
])

discriminator_optimizer = optim.Adam(
    params=discriminator.parameters(),
    lr=discriminator_learning_rate,
    betas=(discriminator_beta1, discriminator_beta2),
    eps=discriminator_epsilon
)

train_dataset = datasets.CIFAR10(
    root=args.dataset_directory,
    train=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    download=True
)

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

test_dataset = datasets.CIFAR10(
    root=args.dataset_directory,
    train=False,
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    download=True
)

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False
)


def create_activation_generator(data_loader):

    inception_v3 = models.inception_v3(pretrained=True)

    for param in inception_v3.parameters():
        param.requires_grad = False

    inception_v3.fc = nn.Identity()

    def activation_generator():

        inception_v3.eval()

        for reals, labels in data_loader:

            with torch.no_grad():

                reals = reals.to(device)
                labels = labels.to(device)

                latents = torch.randn(reals.shape[0], latent_size).to(device)
                latents = mapping_network(latents, labels)
                fakes = generator(latents)

                reals = nn.functional.interpolate(
                    input=reals,
                    size=(299, 299),
                    mode="bilinear"
                )
                fakes = nn.functional.interpolate(
                    input=fakes,
                    size=(299, 299),
                    mode="bilinear"
                )

                real_activations = inception_v3(reals)
                fake_activations = inception_v3(fakes)

                yield real_activations, fake_activations

    return activation_generator


def unnormalize(images, mean, std):

    std = torch.Tensor(std).to(images.device)
    images *= std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mean = torch.Tensor(mean).to(images.device)
    images += mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    return images


summary_writer = SummaryWriter(args.event_directory)
global_step = 0

for epoch in range(args.num_epochs):

    for step, (reals, labels) in enumerate(train_data_loader):

        reals = reals.to(device)
        labels = labels.to(device)

        reals.requires_grad_(True)
        real_logits = discriminator(reals, labels)

        with torch.no_grad():
            latents = torch.randn(reals.size(0), latent_size).to(device)
            latents = mapping_network(latents, labels)
            fakes = generator(latents)

        fakes.requires_grad_(True)
        fake_logits = discriminator(fakes, labels)

        real_losses = nn.functional.softplus(-real_logits)
        fake_losses = nn.functional.softplus(fake_logits)
        discriminator_losses = real_losses + fake_losses

        if real_gradient_penalty_weight:
            real_gradients = torch.autograd.grad(
                outputs=real_logits,
                inputs=reals,
                grad_outputs=torch.ones_like(real_logits),
                retain_graph=True,
                create_graph=True
            )[0]
            real_gradient_penalties = torch.sum(real_gradients ** 2, dim=(1, 2, 3))
            discriminator_losses += real_gradient_penalties * real_gradient_penalty_weight

        if fake_gradient_penalty_weight:
            fake_gradients = torch.autograd.grad(
                outputs=fake_logits,
                inputs=fakes,
                grad_outputs=torch.ones_like(fake_logits),
                retain_graph=True,
                create_graph=True
            )[0]
            fake_gradient_penalties = torch.sum(fake_gradients ** 2, dim=(1, 2, 3))
            discriminator_losses += fake_gradient_penalties * fake_gradient_penalty_weight

        discriminator_loss = torch.mean(discriminator_losses)
        discriminator.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        latents = torch.randn(reals.size(0), latent_size).to(device)
        latents = mapping_network(latents, labels)
        fakes = generator(latents)
        fake_logits = discriminator(fakes, labels)

        fake_losses = nn.functional.softplus(-fake_logits)
        generator_losses = fake_losses

        generator_loss = torch.mean(generator_losses)
        generator.zero_grad()
        generator_loss.backward(retain_graph=False)
        generator_optimizer.step()

        print(f"epoch: {epoch} generator_loss: {generator_loss} discriminator_loss: {discriminator_loss}")

        if step % 100 == 0:

            with torch.no_grad():
                reals = unnormalize(reals, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                fakes = unnormalize(fakes, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

            summary_writer.add_images(
                main_tag="images",
                tag_images_dict=dict(
                    reals=reals,
                    fakes=fakes
                ),
                global_step=global_step
            )
            summary_writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict=dict(
                    generator=generator_loss,
                    discriminator=discriminator_loss
                ),
                global_step=global_step
            )

        global_step += 1

    torch.save(generator.state_dict(), f"{args.checkpoint_directory}/generator/epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{args.checkpoint_directory}/discriminator/epoch_{epoch}.pth")

real_activations, fake_activations = map(torch.cat, zip(*create_activation_generator(test_data_loader)()))
frechet_inception_distance = metrics.frechet_inception_distance(real_activations.numpy(), fake_activations.numpy())


summary_writer.add_scalars(
    main_tag="metrics",
    tag_scalar_dict=dict(
        frechet_inception_distance=frechet_inception_distance
    ),
    global_step=global_step
)

summary_writer.export_scalars_to_json(f"events/scalars.json")
summary_writer.close()

print("----------------------------------------------------------------")
print(f"frechet_inception_distance: {frechet_inception_distance}")
print("----------------------------------------------------------------")
