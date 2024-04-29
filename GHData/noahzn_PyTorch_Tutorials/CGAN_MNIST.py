import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

transform = transforms.Compose([transforms.ToTensor(),
                                ])


train_set = torchvision.datasets.MNIST(root='./data/',
                                       train=True,
                                       transform=transform,
                                       download=False,)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=40,
                                           shuffle=True,
                                           )

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.fc1 = nn.Linear(794, 128)
        #self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.fc1 = nn.Linear(110, 128)
        #self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(128, 784)

    def forward(self, x, y):

        x = torch.cat([x, y], 1)
        input(x)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


def denorm(x):
    return (x + 1) / 2


def one_hot(target):
    y = torch.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y

Generator = G()
Discriminator = D()
Generator.cuda()
Discriminator.cuda()

criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(Generator.parameters(), lr=0.001)
D_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=0.001)


for epoch in range(100):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable((images.view(images.size(0), -1)).cuda())
        labels = Variable(one_hot(labels).cuda())

        real_labels = Variable(torch.ones(images.size(0)).cuda())
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())


        Discriminator.zero_grad()

        noise = Variable(torch.randn(images.size(0), 100).cuda())


        fake_images = Generator(noise, labels).detach()
        real_score = Discriminator(images, labels)
        fake_score = Discriminator(fake_images, labels)

        #real_loss = criterion(real_score, real_labels)
        #fake_loss = criterion(fake_score, fake_labels)
        real_loss = F.binary_cross_entropy(real_score, real_labels)
        fake_loss = F.binary_cross_entropy(fake_score, fake_labels)

        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optimizer.step()


        Generator.zero_grad()
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = Generator(noise, labels)
        outputs = Discriminator(fake_images, labels)
        #G_loss = criterion()
        G_loss = F.binary_cross_entropy(outputs, real_labels)
        G_loss.backward()
        G_optimizer.step()



    print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                  'D(x): %.2f, D(G(z)): %.2f'
                  % (epoch, 50, i + 1, 1500, D_loss.data[0], G_loss.data[0],
                     real_score.data.mean(), fake_score.cpu().data.mean()))


    labels = (torch.ones(40) * 5).long()
    labels = Variable(one_hot(labels)).cuda()
    fake_images = Generator(noise, labels).data[:40]

    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(fake_images,
                                     './data/fake_samples_%d.png' % (epoch))

torch.save(Generator.state_dict(), './generator.pkl')
torch.save(Discriminator.state_dict(), './discriminator.pkl')







