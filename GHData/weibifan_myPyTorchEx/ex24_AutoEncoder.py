# -*- coding: utf-8 -*-
# weibifan 2022-10-30
#  

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

'''
https://cloud.tencent.com/developer/article/2032595
使用.cuda()函数，为什么没用.to(device)函数？

Variable()
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 加载数据集
def get_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 28*28),
                                     nn.Tanh())
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

if __name__ == "__main__":
    # 超参数设置
    batch_size = 128
    lr = 1e-2
    weight_decay = 1e-5
    epoches = 40
    model = autoencoder().to(device)
    # x = Variable(torch.randn(1, 28*28))
    # encode, decode = model(x)
    # print(encode.shape)
    train_data = get_data()
    criterion = nn.MSELoss()
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # weibifan
    if torch.cuda.is_available():
        model.cuda()
        
    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1
        for img, _ in train_data:
            img = img.view(img.size(0), -1)
            img = Variable(img)
            # forward
            _, output = model(img)
            loss = criterion(output, img)
            # backward
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        print("epoch=", epoch, loss.data.float())
        for param_group in optimizier.param_groups:
            print(param_group['lr'])
        if (epoch+1) % 5 == 0:
            print("epoch: {}, loss is {}".format((epoch+1), loss.data))
            pic = to_img(output.cpu().data)
            if not os.path.exists('./simple_autoencoder'):
                os.mkdir('./simple_autoencoder')
            save_image(pic, './simple_autoencoder/image_{}.png'.format(epoch + 1))
    # torch.save(model, './autoencoder.pth')
    # model = torch.load('./autoencoder.pth')
    code = Variable(torch.FloatTensor([[1.19, -3.36, 2.06]]))
    decode = model.decoder(code)
    decode_img = to_img(decode).squeeze()
    decode_img = decode_img.data.cpu().numpy() * 255
    plt.imshow(decode_img.astype('uint8'), cmap='gray')
    plt.show()