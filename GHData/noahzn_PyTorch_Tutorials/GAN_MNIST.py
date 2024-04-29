import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data
import torch.optim


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_set = torchvision.datasets.MNIST(root='./data/',
                                       train=True,
                                       transform=transform,
                                       download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=40,
                                           shuffle=True,
                                           num_workers=2)

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.fc1 = nn.Linear(784, 240)
        self.fc2 = nn.Linear(240, 240)
        self.fc3 = nn.Linear(240, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 784)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        return x

def denorm(x):
    return (x + 1) / 2

Generator = G()
Discriminator = D()
Generator.cuda()
Discriminator.cuda()

criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(Generator.parameters(), lr=0.0005)
D_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=0.0005)


for epoch in range(200):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable((images.view(images.size(0), -1)).cuda())
        real_labels = Variable(torch.ones(images.size(0)).cuda())
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())

        Discriminator.zero_grad()
        outputs = Discriminator(images)
        real_loss = criterion(outputs, real_labels)
        real_score = outputs

        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = Generator(noise)
        outputs = Discriminator(fake_images.detach())  # Returns a new Variable, detached from the current graph.
        fake_loss = criterion(outputs, fake_labels)
        fake_score = outputs

        D_loss = real_loss + fake_loss
        D_loss.backward()
        D_optimizer.step()


        Generator.zero_grad()
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = Generator(noise)
        outputs = Discriminator(fake_images)
        G_loss = criterion(outputs, real_labels)
        G_loss.backward()
        G_optimizer.step()

        if (i + 1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                  'D(x): %.2f, D(G(z)): %.2f'
                  % (epoch, 50, i + 1, 600, D_loss.data[0], G_loss.data[0],
                     real_score.data.mean(), fake_score.cpu().data.mean()))

    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(fake_images.data,
                                     './data/fake_samples_%d.png' % (epoch))

torch.save(Generator.state_dict(), './generator.pkl')
torch.save(Discriminator.state_dict(), './discriminator.pkl')







