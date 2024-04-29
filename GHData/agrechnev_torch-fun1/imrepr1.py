# Created by  IT-JIM  2021
# Here I try to represent an image with a neural net

import sys

import tqdm

import numpy as np
import cv2 as cv
import torch
import torch.utils.data
import torchvision

path_img = '/home/seymour/qwerty/pics_olddesk/absolut.jpg'
batch_size = 8192
n_epoch = 200
device = 'cuda:0'


########################################################################################################################
def print_it(a: np.ndarray, name: str = ''):
    print(name, a.shape, a.dtype, a.min(), a.mean(), a.max())


########################################################################################################################
class ImgDSet(torch.utils.data.Dataset):
    def __init__(self, img_path: str):
        self.img_bgr = cv.imread(img_path)
        self.im_h, self.im_w, _ = self.img_bgr.shape
        img_rgb = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2RGB)
        tran = torchvision.transforms.ToTensor()
        self.img_t = tran(img_rgb)  # (3, H, W)

    def __len__(self):
        return self.im_h * self.im_w

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.im_h * self.im_w + idx

        h, w = self.im_h, self.im_w
        ix, iy = idx % w, idx // w
        x = 2 * ix / w - 1
        y = 2 * iy / h - 1
        tx = torch.tensor([x, y], dtype=torch.float32)
        return tx, self.img_t[:, iy, ix]


########################################################################################################################
class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = torch.nn.Linear(2, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


########################################################################################################################
class Model2(torch.nn.Module):
    def __init__(self, nl: int, nf: int):
        super(Model2, self).__init__()
        self.mlist = torch.nn.ModuleList()
        for i in range(nl):
            n1 = 2 if i == 0 else nf
            n2 = 3 if i == nl - 1 else nf
            self.mlist.append(torch.nn.Linear(n1, n2))

    def forward(self, x):
        for i, m in enumerate(self.mlist):
            x = m(x)
            if i == len(self.mlist) - 1:
                x = torch.sigmoid(x)
            else:
                x = torch.relu(x)
        return x


########################################################################################################################
class Trainer:
    def __init__(self, img_path: str):
        self.dset = ImgDSet(path_img)
        self.dloader_train = torch.utils.data.DataLoader(self.dset, batch_size=batch_size, pin_memory=True,
                                                         num_workers=3, shuffle=True)
        self.dloader_test = torch.utils.data.DataLoader(self.dset, batch_size=batch_size, pin_memory=True,
                                                        num_workers=3, shuffle=False)
        print(f'len_dset = {len(self.dset)}, len_loader = {len(self.dloader_train)}')
        self.model = Model2(10, 50).to(device=device)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()

    def train(self):
        for i_epoch in range(n_epoch):
            losses = []
            for xb, yb in tqdm.tqdm(self.dloader_train):
                xb, yb = xb.to(device), yb.to(device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            print(f'EPOCH {i_epoch}: LOSS={np.mean(losses)}')

    def vis(self) -> np.ndarray:
        """Try to reconstruct the image from model"""
        img = np.zeros_like(self.dset.img_bgr)
        h, w = self.dset.im_h, self.dset.im_w
        with torch.no_grad():
            for xb, yb in tqdm.tqdm(self.dloader_test):
                for x in xb:
                    ix = int(round((x[0].item() + 1) * w / 2))
                    iy = int(round((x[1].item() + 1) * h / 2))
                    y = self.model(x.to(device)).cpu()
                    c = (y.numpy() * 255).astype('uint8')[::-1]
                    img[iy, ix, :] = c
        return img


########################################################################################################################
def main():
    trainer = Trainer(img_path=path_img)
    trainer.train()
    print('Rendering ...')
    img = trainer.vis()
    cv.imshow('img', img)
    cv.waitKey(0)


########################################################################################################################
if __name__ == '__main__':
    main()
