import torch
import torch.tensor as tensor
import numpy as np
import torchvision
import torch.utils.data as Data
from PIL import Image
import matplotlib.pyplot as plt

BATCH_SIZE=16
DOWNLOAD=False

train_data=torchvision.datasets.MNIST(root='mnist_torch/',
                                      train=True,
                                      transform=torchvision.transforms.ToTensor(),
                                      download=DOWNLOAD)
train_load=Data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)


class Discriminator(torch.nn.Module):
    def __init__(self,n_features,n_layer1,n_output):
        super(Discriminator,self).__init__()
        self.layer1 = torch.nn.Linear(n_features, n_layer1)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(n_layer1, n_output)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        layer1 = self.layer1(x)
        relu = self.relu(layer1)
        layer2 = self.layer2(relu)
        return self.sigmoid(layer2),layer2

class Generator(torch.nn.Module):
    def __init__(self,n_features,n_layer1,n_output):
        super(Generator,self).__init__()
        self.layer1=torch.nn.Linear(n_features,n_layer1)
        self.relu=torch.nn.ReLU()
        self.layer2=torch.nn.Linear(n_layer1,n_output)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, x):
        layer1=self.layer1(x)
        relu=self.relu(layer1)
        layer2=self.layer2(relu)
        return self.sigmoid(layer2)

def sample_Z(m, n):  # 生成维度为[m, n]的随机噪声作为生成器G的输入
    return np.random.uniform(-1., 1., size=[m, n])

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

d=Discriminator(784,128,1)
g=Generator(100,128,784)
optimizer_g=torch.optim.Adam(g.parameters(),lr=0.001)
optimizer_d=torch.optim.Adam(d.parameters(),lr=0.001)

loss=torch.nn.BCELoss(reduction='none')

plt.ion()

for i,(x,y) in enumerate(train_load):
    z = sample_Z(BATCH_SIZE, 100)
    z = tensor(z.astype(np.float32))
    g_sample = g(z)
    D_fake, D_logit_fake = d(g_sample)

    x = torch.reshape(x, (BATCH_SIZE, 1, -1))
    D_real, D_logit_real=d(x)

    D_loss_real=torch.mean(loss(D_real,torch.ones(BATCH_SIZE,1)))
    D_loss_fake=torch.mean(loss(D_fake,torch.zeros(BATCH_SIZE,1)))
    D_loss=D_loss_real+D_loss_fake

    G_loss=torch.mean(loss(D_fake,torch.ones(BATCH_SIZE,1)))

    optimizer_g.zero_grad()
    G_loss.backward(retain_graph=True)
    optimizer_g.step()

    optimizer_d.zero_grad()
    D_loss.backward()
    optimizer_d.step()
    if (i+1)%50==0:
        z = sample_Z(1, 100)
        z = tensor(z.astype(np.float32))
        g_sample = g(z)
        img=np.reshape(g_sample.data.numpy(),(28,28))
        plt.cla()
        plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()









