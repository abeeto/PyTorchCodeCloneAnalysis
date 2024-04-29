import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.image as gImage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 64, 6, 6)

class VAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=32):
        super(VAE, self).__init__()

        h_dim = 2304

        # (batch_size, ch, h, w)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),    # torch.Size([32, 2304, 1, 1])
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=image_channels, kernel_size=3, stride=2),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print('h.shape', h.shape)     # h.shape torch.Size([32, 2304])
        z, mu, logvar = self.bottleneck(h)
        # print('z.shape', z.shape)     # z.shape torch.Size([32, 16])
        return z, mu, logvar

    def decode(self, z):
        a = self.fc3(z)
        # print('a.shape', a.shape)  # a.shape torch.Size([32, 2304])
        x = self.decoder(a)
        # print('x.shape', x.shape)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

class VAE_trainer(object):
    def __init__(self, image_channels=3, z_dim=32, batch_size=64, epoch=50):
        self.batch_size = batch_size
        self.epoch = epoch
        self.image_channels = image_channels
        self.z_dim = z_dim
        self.vae = VAE(image_channels, z_dim)
        print('establish VAE object')

    def reset_model(self):
        self.vae = VAE(self.image_channels, self.z_dim)

    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

    def list_to_batches(self, x):
        batch_num = math.ceil(len(x) / self.batch_size)
        batch_list = []  # [batch_size, 3, 15, 15]
        for k in range(batch_num):
            first_img = self.img_normalization(x[int(k*self.batch_size)])
            temp_batch = self.img_to_tensor(first_img)
            temp_batch = temp_batch.unsqueeze(0)

            for i in range(1, self.batch_size):
                if int(k*self.batch_size+i)<len(x):
                    img = self.img_normalization(x[int(k*self.batch_size+i)])   # array(15,15,3)
                    img = self.img_to_tensor(img)
                    img = img.unsqueeze(0)
                    temp_batch = torch.cat([temp_batch, img], dim=0)
            # print('temp_batch', temp_batch.shape)
            batch_list.append(temp_batch)
        return batch_list

    def img_normalization(self, img):
        new_img = img.copy()
        if np.max(np.max(np.max(new_img)))>1.0:
            new_img = new_img/255
        return new_img

    def img_to_tensor(self, img):
        img_tensor = torch.Tensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def train(self, x):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        batch_num = math.ceil(len(x) / self.batch_size)
        batch_list = self.list_to_batches(x)
        for epoch in range(self.epoch):
            print('training epoch', epoch)
            for k in range(batch_num):
                images_batch = batch_list[k]
                # print(images_batch.shape)
                recon_images, mu, logvar = self.vae(images_batch)
                loss, bce, kld = self.loss_fn(recon_images, images_batch, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch', epoch, 'loss', loss.data.item()/self.batch_size)

    def encode(self, x):
        x = self.img_normalization(x)
        x = self.img_to_tensor(x)
        x = x.unsqueeze(0)
        z, mu, logvar = self.vae.encode(x)
        return z

    def decode(self, z):
        x = self.vae.decode(z)
        x = x.squeeze()
        x = x.permute(1, 2, 0)
        x = x.detach().numpy()
        return x

    def forward(self, x):
        pred_x = self.decode(self.encode(x))
        return pred_x

    def save_model(self):
        torch.save(self.vae, 'vae.pkl')

    def load_model(self, name):
        self.vae = torch.load(name)

if __name__ == '__main__':
    image_list = []
    for i in range(800):
        image = gImage.imread('./pig_pic/'+str(i)+'.bmp')
        image_list.append(image)
    print('loaded', len(image_list), 'images')
    # plt.imshow(image_list[0])
    # plt.show()

    my_vae = VAE_trainer(image_channels=3, z_dim=16, batch_size=32, epoch=500)

    my_vae.train(image_list)

    my_vae.save_model()

    #my_vae.load_model('vae.pkl')

    for i in range(100):
        print('prediction of image', i)
        z = my_vae.encode(image_list[i])
        print('latent variable is', z)
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.imshow(image_list[i])
        x = my_vae.decode(z)
        ax2.imshow(x)
        plt.show()

