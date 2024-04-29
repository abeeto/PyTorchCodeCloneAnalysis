import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim
import numpy as np


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.fc1 = nn.Linear(74, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 7*7*128)
        self.fc2_bn = nn.BatchNorm1d(7*7*128)

        self.cov1 = nn.Conv2d(128, 64, kernel_size=4, stride=1, padding=1, bias=False)
        self.cov1_bn = nn.BatchNorm2d(64)

        self.cov2 = nn.ConvTranspose2d(64, 1, kernel_size=4)

    def forward(self, x):
        print('a')
        x = F.relu(self.fc1_bn(self.fc1(x)))
        print('b')
        x = F.relu(self.fc2_bn(self.fc2(x)))
        print('c')
        x = F.relu(self.cov1_bn(self.cov1(x)))
        print('d')
        x = F.sigmoid(self.cov2(x))

        return x


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.cov1 = nn.Conv2d(1, 64, kernel_size=4, stride=2)

        self.cov2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cov2_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(7*7*128, 1024)
        self.fc1_bn = nn.BatchNorm2d(1024)

        self.fc_d = nn.Linear(1024, 2)

        self.fc_mi1 = nn.Linear(1024, 128)
        self.fc_mi1_bn = nn.BatchNorm2d(128)

        self.fc_mi2 = nn.Linear(128, 2+10)


    def forward(self, x):
        x = F.leaky_relu(self.cov1(x), 0.1)
        x = F.leaky_relu(self.cov2_bn(self.cov2(x)), 0.1)
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)), 0.1)

        d = self.fc_d(x)

        mi = F.leaky_relu(self.fc_mi1_bn(self.fc_mi1(x)), 0.1)
        mi = self.fc_mi2(mi)

        return d, mi


def one_hot(target):
    y = torch.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y

train_set = torchvision.datasets.MNIST(root='./data/',
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)

def gen_discrete_code(n_instance, n_discrete, num_category=10):
    """generate discrete codes with n categories"""
    codes = []
    for i in range(n_discrete):
        code = np.zeros((n_instance, num_category))
        random_cate = np.random.randint(0, num_category, n_instance)
        code[range(n_instance), random_cate] = 1
        codes.append(code)

    codes = np.concatenate(codes, 1)
    return torch.Tensor(codes)

def rnd_categorical(n, n_categorical):
    indices = np.random.randint(n_categorical, size=n)
    one_hot = np.zeros((n, n_categorical))
    one_hot[np.arange(n), indices] = 1
    return one_hot, indices

generator = G()
discriminator = D()
generator.cuda()
discriminator.cuda()

criterion = nn.CrossEntropyLoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

for epoch in range(50):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable((images.view(images.size(0), -1)).cuda())
        labels = Variable(one_hot(labels).cuda())
        real_labels = Variable(torch.ones(images.size(0)).cuda())
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())

        discriminator.zero_grad()


        noise_ = np.random.uniform(-1, 1, (images.size(0), 62))

        continuous_noise = np.random.randn(images.size(0), 2) * 1 + 0

        #discrete_noise = gen_discrete_code(10, 1)

        discrete_noise, discrete_noise_indice = rnd_categorical(images.size(0), 10)
        discrete_noise_indice = Variable(torch.from_numpy(np.asarray(discrete_noise_indice, dtype=np.int32)
                                                          )).cuda()
        # discrete_noise = Variable(torch.from_numpy(discrete_noise)).cuda()



        noise = np.concatenate((noise_, continuous_noise, discrete_noise), axis=1)

        fake_images = generator(Variable(torch.FloatTensor(noise)).cuda())
        d_fake, mi_fake = discriminator(fake_images)
        input(images.size(2))
        d_real, mi_real = discriminator(images)

        x_real = np.zeros((images.size(0), images.size(2)), dtype=np.float32)
        # for xi in range(len(x_real)):
        #     x_real[xi] = np.array(images[np.random.randint(train_size)])
        x_real = np.expand_dims(x_real, 1)
        y_real, _ = discriminator(x_real)








