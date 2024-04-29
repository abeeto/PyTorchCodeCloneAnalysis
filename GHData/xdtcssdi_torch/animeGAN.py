import torch as t
from torch import nn
import torchvision as tv
from tqdm import tqdm
 

class Base(nn.Module):

    def block(self, feature_in, feature_out, kernel, stride, padding, leak=0.2, transpose=False, last=False):
        if not transpose:
            if last:
                return nn.Conv2d(feature_in, feature_out, kernel, stride, padding, bias=False)
            else:
                return nn.Sequential(
                    nn.Conv2d(feature_in, feature_out, kernel, stride, padding, bias=False),
                    nn.BatchNorm2d(feature_out),
                    nn.LeakyReLU(leak, inplace=True)
                )
        else:
            if last:
                return nn.ConvTranspose2d(feature_in, feature_out, kernel, stride, padding, bias=False)
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(feature_in, feature_out, kernel, stride, padding, bias=False),
                    nn.BatchNorm2d(feature_out),
                    nn.ReLU(inplace=True)
                )


class G(Base):
    def __init__(self):
        super(G, self).__init__()
        ng = 64
        bk1 = self.block(100, ng * 8, 4, 1, 0, leak=0.2, transpose=True)
        bk2 = self.block(ng * 8, ng * 4, 4, 2, 1, leak=0.2, transpose=True)
        bk3 = self.block(ng * 4, ng * 2, 4, 2, 1, leak=0.2, transpose=True)
        bk4 = self.block(ng * 2, ng, 4, 2, 1, leak=0.2, transpose=True)
        bk5 = self.block(ng, 3, 5, 3, 1, leak=0.2, transpose=True, last=True)
        self.model = nn.Sequential(bk1, bk2, bk3, bk4, bk5, nn.Tanh())

    def forward(self, x):
        return self.model(x)


class D(Base):
    def __init__(self):
        super(D, self).__init__()
        nd = 64
        bk1 = self.block(3, nd, 5, 3, 1, transpose=False)
        bk2 = self.block(nd, nd * 2, 4, 2, 1, transpose=False)
        bk3 = self.block(nd * 2, nd * 4, 4, 2, 1, transpose=False)
        bk4 = self.block(nd * 4, nd * 8, 4, 2, 1, transpose=False)
        bk5 = self.block(nd * 8, 1, 4, 1, 0, transpose=False, last=True)
        self.model = nn.Sequential(bk1, bk2, bk3, bk4, bk5, nn.Sigmoid())

    def forward(self, x):
        return self.model(x).view(-1)




def createDatabases():
    img_dir = "/home/oswin/datasets"
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(96),
        tv.transforms.CenterCrop(96),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = tv.datasets.ImageFolder(img_dir, transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=256,
                                         shuffle=True,
                                         num_workers=4,
                                         drop_last=True
                                         )
    return dataloader


def train():
    gpu = True
    device = t.device('cuda') if gpu else t.device('cpu')
    epochs = 1000
    batch_size = 256
    noise_dim = 100
    dataloader = createDatabases()

    d = D().to(device)
    g = G().to(device)

    # 损失函数
    criterion = t.nn.BCELoss().to(device)
    # 用于对比损失函数的标签
    true_labels = t.ones(batch_size).to(device)
    fake_labels = t.zeros(batch_size).to(device)

    # 优化器
    optimizer_d = t.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_g = t.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    for epoch in range(1, epochs + 1):
        loss_d, loss_g = None,None
        for i, (img, _) in tqdm(enumerate(dataloader)):
            img = img.to(device)

            if i % 1 == 0:
                # 判别器
                optimizer_d.zero_grad()
                output = d(img)  # 判别真实图片
                error_d_true = criterion(output, true_labels)  # 让真图片尽量变成真
                error_d_true.backward()  # 反向传播
                noise = t.randn(batch_size, noise_dim, 1, 1).to(device)  # 生成噪声
                fake_img = g(noise).detach()  # 生成图片，且从计算图剥离
                output = d(fake_img)  # 判别假图片
                error_d_fake = criterion(output, fake_labels)  # 让假图片尽量变成假
                error_d_fake.backward()
                optimizer_d.step()
                loss_d = (error_d_true + error_d_fake).item()

            if i % 5 == 0:
                # 生成器
                optimizer_g.zero_grad()
                noise = t.randn(batch_size, noise_dim, 1, 1).to(device)
                fake_img = g(noise)  # 生成假图片
                output = d(fake_img)  # 判别假图片
                error_g = criterion(output, true_labels)  # 让假图片尽量变成真图片
                error_g.backward()
                optimizer_g.step()
                loss_g = error_g.item()

        # 保存图片
        tv.utils.save_image(fake_img.data[:64], f"result{epoch}.jpg", normalize=True, range=(-1, 1))
        
        print(epoch, loss_d, loss_g)
train()