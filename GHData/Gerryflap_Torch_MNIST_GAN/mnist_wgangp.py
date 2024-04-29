import torch
import torch.nn.functional as F


def mish(x):
    # Mish activation https://arxiv.org/pdf/1908.08681v2.pdf
    return x * torch.tanh(F.softplus(x))


class MnistGenerator(torch.nn.Module):
    def __init__(self, latent_size, h_size, use_sine=False, use_mish=False, bias=True):
        super().__init__()
        self.latent_size = latent_size
        self.h_size = h_size
        self.use_sine = use_sine
        self.bias = bias
        if use_sine:
            self.activ = torch.sin
        elif use_mish:
            self.activ = mish
        else:
            self.activ = torch.relu

        self.conv_1 = torch.nn.ConvTranspose2d(self.latent_size, self.h_size * 4, 4, bias=self.bias)
        self.conv_2 = torch.nn.ConvTranspose2d(self.h_size * 4, self.h_size * 2, kernel_size=5, stride=2, bias=self.bias)
        self.conv_3 = torch.nn.ConvTranspose2d(self.h_size * 2, self.h_size, kernel_size=5, stride=2, bias=self.bias)
        self.conv_4 = torch.nn.ConvTranspose2d(self.h_size, 1, kernel_size=4, stride=1, bias=self.bias)

        self.bn_1 = torch.nn.BatchNorm2d(self.h_size * 4)
        self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
        self.bn_3 = torch.nn.BatchNorm2d(self.h_size)

    def forward(self, inp):
        x = inp.view(-1, self.latent_size, 1, 1)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activ(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        x = torch.tanh(x) if not self.use_sine else torch.sin(x)

        return x

    def generate_z_batch(self, batch_size):
        z = torch.normal(torch.zeros((batch_size, self.latent_size)), 1)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        return z

    def generate_batch(self, batch_size):
        # Generate random latent vectors
        z = self.generate_z_batch(batch_size)

        # Return outputs
        return self(z)


class MnistDiscriminator(torch.nn.Module):
    def __init__(self, h_size, use_bn=False, use_sine=False, use_mish=False):
        super().__init__()

        self.use_sine = use_sine
        if use_sine:
            self.activ = torch.sin
        elif use_mish:
            self.activ = mish
        else:
            self.activ = self.leaky_relu

        self.h_size = h_size
        self.conv_1 = torch.nn.Conv2d(1, h_size, kernel_size=4,  stride=1)
        self.conv_2 = torch.nn.Conv2d(h_size, h_size * 2, kernel_size=5, stride=2)
        self.conv_3 = torch.nn.Conv2d(h_size * 2, h_size * 4, kernel_size=5, stride=2)
        self.conv_4 = torch.nn.Conv2d(h_size * 4, h_size * 4, kernel_size=4, stride=1)

        self.use_bn = use_bn
        if use_bn:
            self.bn_2 = torch.nn.BatchNorm2d(self.h_size * 2)
            self.bn_3 = torch.nn.BatchNorm2d(self.h_size * 4)
            self.bn_4 = torch.nn.BatchNorm2d(self.h_size * 4)

        self.lin_1 = torch.nn.Linear(h_size * 4, h_size * 4)
        self.lin_2 = torch.nn.Linear(h_size * 4, 1)

    @staticmethod
    def leaky_relu(x):
        return F.leaky_relu(x, 0.02)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.activ(x)

        x = self.conv_2(x)
        if self.use_bn:
            x = self.bn_2(x)
        x = self.activ(x)

        x = self.conv_3(x)
        if self.use_bn:
            x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        if self.use_bn:
            x = self.bn_4(x)
        x = self.activ(x)

        # Flatten to vector
        x = x.view(-1, self.h_size * 4)

        x = self.lin_1(x)
        x = self.activ(x)

        x = self.lin_2(x)
        return x

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MNIST WGAN-GP application.")
    parser.add_argument("--live_view", action="store_true", default=False, help="Adds a Matplotlib live view that shows samples")
    parser.add_argument("--batch_size", action="store", type=int, default=64, help="Changes the batch size, default is 64")
    parser.add_argument("--lr", action="store", type=float, default=0.0001, help="Changes the learning rate, default is 0.0001")
    parser.add_argument("--h_size", action="store", type=int, default=16, help="Sets the h_size, which changes the size of the network")
    parser.add_argument("--epochs", action="store", type=int, default=100, help="Sets the number of training epochs")
    parser.add_argument("--d_steps", action="store", type=int, default=2, help="Amount of discriminator steps per generator step")
    parser.add_argument("--l_size", action="store", type=int, default=12, help="Size of the latent space")
    parser.add_argument("--print_steps", action="store", type=int, default=50, help="Number of generator steps between prints/live view updates")
    parser.add_argument("--load_path", action="store", type=str, default=None, help="When given, loads models from LOAD_PATH folder")
    parser.add_argument("--save_path", action="store", type=str, default=None, help="When given, saves models to LOAD_PATH folder after all epochs (or every epoch)")
    parser.add_argument("--save_every_epoch", action="store_true", default=False, help="When a save path is given, store the model after every epoch instead of only the last")
    parser.add_argument("--cuda", action="store_true", default=False, help="Enables CUDA support. The script will fail if cuda is not available")
    parser.add_argument("--use_sine", action="store_true", default=False, help="Changes all activations except the ouput of D to sin(x), which has interesting effects")
    parser.add_argument("--use_mish", action="store_true", default=False, help="Changes all activations except the ouput of D and G to mish, which might work better")
    parser.add_argument("--img_path", action="store", type=str, default=None, help="When given, saves samples to the given directory")
    parser.add_argument("--no_bias_in_G", action="store_true", default=False, help="Disables biases in the Generator")


    args = parser.parse_args()

    from torchvision import datasets
    from torchvision import transforms
    import numpy as np

    live_view = args.live_view
    if live_view or args.img_path:
        import matplotlib.pyplot as plt

    batch_size = args.batch_size
    learning_rate = args.lr

    # Number of channels of the highest resolution convolutional layer of both networks.
    # Deeper layers are scaled by multiples of this number
    h_size = args.h_size

    epochs = args.epochs
    latent_size = args.l_size

    # Number of steps taken by the discriminator for each update to the generator
    n_d_steps = args.d_steps

    dataset = datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda img: img*2 - 1)
                       ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    if args.load_path is None:
        generator = MnistGenerator(latent_size=latent_size, h_size=h_size, use_sine=args.use_sine, use_mish=args.use_mish, bias=not args.no_bias_in_G)
        discriminator = MnistDiscriminator(h_size=h_size, use_bn=False, use_sine=args.use_sine, use_mish=args.use_mish)
    else:
        generator = torch.load(args.load_path + "generator.pt", map_location=torch.device('cpu'))
        discriminator = torch.load(args.load_path + "discriminator.pt", map_location=torch.device('cpu'))



    optim_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    def save_models(path):
        torch.save(generator, path+"generator.pt")
        torch.save(discriminator, path+"discriminator.pt")
        torch.save(optim_G.state_dict(), path+"optim_G.pt")
        torch.save(optim_D.state_dict(), path+"optim_D.pt")

    if args.load_path is not None:
        optim_G.load_state_dict(torch.load(args.load_path + "optim_G.pt", map_location=torch.device('cpu')))
        optim_D.load_state_dict(torch.load(args.load_path + "optim_D.pt", map_location=torch.device('cpu')))
        print("Warning: the learning rate of the loaded optimizers will override the value given to this script")

    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()


    if live_view:
        plt.ioff()

    test_zs = generator.generate_z_batch(8)
    for epoch in range(epochs):
        for i, (real_batch, _) in enumerate(dataloader):
            if real_batch.size()[0] != batch_size:
                continue
            if i%args.d_steps == 0:
                # Train G (this is sometimes skipped to balance G and D according to the d_steps parameter)

                # Make gradients for G zero
                optim_G.zero_grad()

                # Put the generator in train mode and discriminator in eval mode. This affects batch normalization
                generator.train()
                discriminator.eval()

                # Generate a batch of fakes
                fake_batch = generator.generate_batch(batch_size)

                # Compute loss for G, images should become more 'real' to the discriminator
                g_loss = -discriminator(fake_batch).mean()
                g_loss.backward()
                optim_G.step()

            # Train D

            # Make gradients for D zero
            optim_D.zero_grad()

            # Put the generator in eval mode and discriminator in train mode. This affects batch normalization
            generator.eval()
            discriminator.train()

            # Generate a fake image batch
            fake_batch = generator.generate_batch(batch_size)

            # Compute outputs for fake images
            d_fake_outputs = discriminator(fake_batch)

            if args.cuda:
                real_batch = real_batch.cuda()

            # Compute outputs for real images
            d_real_outputs = discriminator(real_batch)

            # Compute losses
            d_loss = (d_fake_outputs - d_real_outputs).mean()

            eps = torch.rand((batch_size, 1, 1, 1))
            if args.cuda:
                eps = eps.cuda()
            x_hat = eps * real_batch + (1.0-eps) * fake_batch
            grad = torch.autograd.grad(discriminator(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]
            d_grad_loss = torch.pow(torch.norm(grad, 2, dim=(1,2,3)) - 1, 2).mean()

            d_loss = d_loss + 10 * d_grad_loss

            # Back propagate
            d_loss.backward()

            # Update weights
            optim_D.step()

            if i%args.print_steps == 0:
                print("Epoch: %d, batch %d/%d"%(epoch, i, len(dataloader)))
                print("G loss: ", g_loss.detach().item())
                print("D loss: ", d_loss.detach().item())
                print()

                if live_view or args.img_path is not None:
                    generator.eval()
                    discriminator.eval()
                    plt.clf()
                    plt.title("Epoch: %d, batch %d/%d"%(epoch, i, len(dataloader)))

                    imgs = generator(test_zs).detach().cpu().numpy()

                    plot_img = np.concatenate(list(imgs), axis=2)[0, :, :]
                    plt.imshow(plot_img, cmap='gray')

                    if live_view:
                        plt.pause(0.001)

                    if args.img_path is not None:
                        plt.savefig(args.img_path + "img_%04d_%04d.png"%(epoch, i))

        if args.save_every_epoch and args.save_path:
            save_models(args.save_path)
    if args.save_path:
        save_models(args.save_path)