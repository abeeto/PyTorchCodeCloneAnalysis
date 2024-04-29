import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Discriminator will be a simple CNN
# Output after Conv2d operations
# output = (N+(2*p) - F)/stride + 1
# output after max_pool2d operations has the same formula as above!


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1440, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # x = (28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x -> conv2d(1, 10) -> x=(24, 24, 10) -> maxpool2d(2) -> x=(12, 12)
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 49*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)

        x = self.ct1(x)
        x = F.relu(x)
        x = self.ct2(x)
        x = F.leaky_relu(x)

        # to 28x28
        return self.conv(x)


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.disc = Discriminator()

        # random noise
        self.validation_z = torch.randn(6, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)

        # train generator: max log(D(G(z)))
        if optimizer_idx == 0:
            fake_imgs = self(z)
            y_hat = self.disc(fake_imgs)
            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(real_imgs)

            g_loss = self.adversarial_loss(y_hat, y)

            log_dict = {"g_loss": g_loss}
            return {"loss": g_loss, "progress_bar": log_dict, "log": log_dict}

        # train disc: max log(D(x)) + log(1 - D(G(z)))
        if optimizer_idx == 1:
            y_hat_real = self.disc(real_imgs)
            y_real = torch.ones(real_imgs.size(0), 1)
            y_real = y_real.type_as(real_imgs)

            real_loss = self.adversarial_loss(y_hat_real, y_real)

            y_hat_fake = self.disc(self(z).detach())
            y_fake = torch.zeros(real_imgs.size(0), 1)
            y_fake = y_fake.type_as(real_imgs)

            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

            d_loss = (real_loss + fake_loss) / 2
            log_dict = {"d_loss": d_loss}
            return {"loss": d_loss, "progres_bar": log_dict, "log": log_dict}

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_disc = torch.optim.Adam(self.disc.parameters(), lr=lr)

        return [opt_generator, opt_disc], []

    def plot_imgs(self):
        # type_as moves to the same device!
        z = self.validation_z.type_as(self.generator.fc1.weight)
        sample_imgs = self(z).cpu()

        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i + 1)
            # plt.tight_layout()
            plt.imshow(
                sample_imgs.detach()[i, 0, :, :], cmap="gray_r", interpolation="none"
            )
            plt.title(f"generated_data at epoch: {self.current_epoch}")
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
        plt.show()
