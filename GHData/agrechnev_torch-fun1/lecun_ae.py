# By IT-JIM, 19-Apr-2021

import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import torch
import torch.utils.data
import torchvision


########################################################################################################################
class AE(torch.nn.Module):
    def __init__(self, d):
        super(AE, self).__init__()
        self.d = d
        im_size = 28 * 28
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(im_size, d),
            torch.nn.Tanh(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(d, im_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


########################################################################################################################
def to_img(x):
    x = x.detach().cpu()
    x = 0.5 * (x + 1)
    x = x.view(x.size(0), 28, 28)
    return x


def display_images0(data, nn, title):
    plt.figure(figsize=(18, max(6, 3 * nn)))
    for n in range(nn):
        for i in range(4):
            plt.subplot(nn, 4, i + 1 + 4 * n)
            plt.imshow(data[i + 4 * n])
            plt.axis('off')
    plt.suptitle(title)


def display_images(in_, out, nn=1, t1='IN', t2='OUT'):
    if in_ is not None:
        in_pic = to_img(in_)
        display_images0(in_pic, nn, t1)
    out_pic = to_img(out)
    display_images0(out_pic, nn, t2)
    plt.show()


########################################################################################################################
class Trainer:
    def __init__(self, denoise=False):
        self.denoise = denoise
        self.n_epochs = 20
        self.batch_size = 256
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Dset : MNIST
        tran = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])
        self.dset = torchvision.datasets.MNIST('~/w/data', download=True, transform=tran)
        self.dloader = torch.utils.data.DataLoader(self.dset, batch_size=self.batch_size, shuffle=True)

        # Model etc
        self.model = AE(500 if denoise else 30).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        if denoise:
            self.do = torch.nn.Dropout()

    def training_loop(self):
        print('Training ...')
        for epoch in range(self.n_epochs):
            for img, _ in self.dloader:
                img = img.to(self.device).view(img.size(0), -1)
                if self.denoise:
                    noise = self.do(torch.ones(img.shape)).to(self.device)
                    img_in = img * noise
                else:
                    img_in = img
                out = self.model(img_in)
                loss = self.criterion(out, img.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch + 1}/ {self.n_epochs}], loss = {loss.item() : .4f}')
        # display_images(img_in, out)

        if self.denoise:
            self.inpaint_fun(noise, img_in, img, out)

    def inpaint_fun(self, noise, img_bad, img, output):
        y_telea, y_ns = [], []
        for i in range(4):
            img_cor = img_bad.detach().cpu()[i].view(28, 28)
            img_cor = ((img_cor / 4 + 0.5) * 255).byte().numpy()
            mask = 2 - noise.cpu()[i].view(28, 28).byte().numpy()
            y_telea.append(cv.inpaint(img_cor, mask, 3, cv.INPAINT_TELEA))
            y_ns.append(cv.inpaint(img_cor, mask, 3, cv.INPAINT_NS))
        y_telea = [torch.from_numpy(x) for x in y_telea]
        y_ns = [torch.from_numpy(x) for x in y_ns]
        y_telea = torch.stack(y_telea).float()
        y_ns = torch.stack(y_ns).float()


        display_images(noise[:4], img_bad[:4], 1, 'NOISE', 'BAD')
        display_images(img[:4], output[:4], 1, 'IMG', 'OUTPUT')
        display_images(y_telea, y_ns, 1, 'TELEA', 'NS')


########################################################################################################################
def main():
    trainer = Trainer(True)
    trainer.training_loop()
    # display_images(None, trainer.model.encoder[0].weight, 5)


########################################################################################################################
if __name__ == '__main__':
    main()
