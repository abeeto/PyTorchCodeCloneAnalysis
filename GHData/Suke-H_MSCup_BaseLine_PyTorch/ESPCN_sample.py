import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import random
from PIL import Image, ImageOps
from tqdm import tqdm

# モデル定義
class ESPCN(nn.Module):
    def __init__(self):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 4**2*3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

# 学習
class trainer(object):
    def __init__(self, device, trained_model=""):
        self.device = device
        self.model = ESPCN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        if trained_model != "":
            self.model.load_weights(trained_model)

    def train(self, lr_imgs, hr_imgs, out_path, batch_size, epochs):
        
        self.model.train()
        device, optimizer, criterion = self.device, self.optimizer, self.criterion
        
        # データセット
        dataset = data.TensorDataset(torch.from_numpy(lr_imgs), torch.from_numpy(hr_imgs))
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) # shuffle=True
        
        # 学習
        for epoch in tqdm(range(1, epochs+1)):
            for _epoch, (lr, hr) in enumerate((dataloader), 1):
                lr, hr = lr.to(device), hr.to(device)

                optimizer.zero_grad()
                re = self.model(lr)
                loss = criterion(re, hr)
                loss.backward()
                optimizer.step()

        print("___Training finished\n\n")

        # パラメータ保存
        print("___Saving parameter...")
        torch.save(self.model.state_dict(), out_path+".pth")
        print("___Successfully completed\n\n")

        return self.model

# Dataset creation
def create_dataset():

    print("\n___Creating a dataset...")
    prc = ['/', '-', '\\', '|']
    cnt = 0
    training_data =[]

    for i in range(60):
        d = "./train/"

        # High-resolution image
        img = Image.open(d+"train_{}_high.tif".format(i))
        flip_img = np.array(ImageOps.flip(img))
        mirror_img = np.array(ImageOps.mirror(img))
        img = np.array(img)

        img = img.astype(np.float32) / 255.0
        flip_img = flip_img.astype(np.float32) / 255.0
        mirror_img = mirror_img.astype(np.float32) / 255.0

        # Low-resolution image
        low_img = Image.open(d+"train_{}_low.tif".format(i))
        low_flip_img = np.array(ImageOps.flip(low_img))
        low_mirror_img = np.array(ImageOps.mirror(low_img))
        
        low_img = np.array(low_img)
        low_img = low_img.astype(np.float32) / 255.0
        low_flip_img = low_flip_img.astype(np.float32) / 255.0
        low_mirror_img = low_mirror_img.astype(np.float32) / 255.0

        training_data.append([img,low_img])
        training_data.append([flip_img,low_flip_img])
        training_data.append([mirror_img,low_mirror_img])

        cnt += 1

        print("\rLoading a LR-images and HR-images...{}    ({} / {})".format(prc[cnt%4], cnt, 60), end='')

    print("\rLoading a LR-images and HR-images...Done    ({} / {})".format(cnt, 60), end='')
    print("\n___Successfully completed\n")

    random.shuffle(training_data)   
    lr_imgs = []
    hr_imgs = []

    for hr, lr in training_data:
        lr_imgs.append(lr)
        hr_imgs.append(hr)
        
    # channel firstにする
    lr_imgs = np.array(lr_imgs).transpose(0, 3, 1, 2)
    hr_imgs = np.array(hr_imgs).transpose(0, 3, 1, 2)

    return lr_imgs, hr_imgs

if __name__ == "__main__":

    # データセットの読み込み
    lr_imgs, hr_imgs = create_dataset()

    # 学習
    print("___Start training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Trainer = trainer(device)
    model = Trainer.train(lr_imgs, hr_imgs, out_path="espcn_model_weight" , batch_size=15, epochs=1400)
    