import torch
import torchvision
import torchvision.datasets as dsets
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import sklearn
import numpy as np
import os

# hyper Parameters

lr = 0.001
input_size = 784
hidden_size = 256
latent_size = 100
output_size = 1
epochs = 35

# Import data

par_dir = './notMNIST'
path = os.listdir(par_dir)
image_data = []
labels = []
batch_size = 128

i = 0

for folder in path:
    
    images = os.listdir(par_dir +'/'+ folder)
    
    for image in images:

        file = par_dir +'/'+ folder +'/'+ image
        
        if(os.path.getsize(file)>0):
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (28, 28))
            image_data.append(img)
        else:
            print('File :' + par_dir+'/'+folder+'/'+image + 'is empty')
        
        labels.append(i)
    
    i += 1

image_train = image_data[:12800]
label_train = labels[:12800]

im, lab = sklearn.utils.shuffle(image_train, label_train)

#image_test = image_data[1645:]
#label_test = labels[1645:]

def get_train_data(input):

    batch_images = image_train[input*batch_size : (input+1)*batch_size]
    batch_labels = label_train[input*batch_size : (input+1)*batch_size]
    batch_labels = np.array(batch_labels, dtype = np.int)

    return batch_images, batch_labels

# Model

# discriminator

class discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, output_size):

        super(discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.r1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.r2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(self.hidden_size, output_size)
        self.r3 = nn.Sigmoid()
        
        self.d = nn.Sequential(self.fc1, self.r1,
                               self.fc2, self.r2,
                               self.fc3, self.r3)

    def forward(self, x):

        x = x.view(batch_size, -1)

        output = self.d(x)

        return output

class generator(nn.Module):

    def __init__(self, latent_size, hidden_size, image_size, batch_size):

        super(generator, self).__init__()

        self.latent_size = latent_size
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.fc1 = nn.Linear(self.latent_size, self.hidden_size)
        self.r1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.r2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(self.hidden_size, self.image_size)
        self.r3 = nn.Tanh()

        self.g = nn.Sequential(self.fc1, self.r1,
                               self.fc2, self.r2,
                               self.fc3, self.r3)

    def forward(self, x):

        x = x.view(self.batch_size, -1)

        output = self.g(x)

        return output


D = discriminator(input_size, hidden_size, batch_size, output_size)

G = generator(latent_size, hidden_size, input_size, batch_size)

criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr = lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr = lr)

def reset_grad():

    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

for epoch in range(epochs):

    for i in range(100):

        im, _ = get_train_data(i)
        im = Variable(torch.Tensor(im))
        im = im.view(batch_size, 1, 28, 28)

        ######################################

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = D(im)
        real = outputs
        d_loss_real = criterion(real, real_labels)

        z = torch.randn(batch_size, latent_size)

        #######################################

        outputs = G(z)

        outputs = D(outputs)
        fake = outputs
        d_loss_fake = criterion(fake, fake_labels)

        ########################################

        d_total_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_total_loss.backward()
        d_optimizer.step()


        #########################################

        z = torch.randn(batch_size, latent_size)

        outputs = G(z)
        gen = outputs

        outputs = D(gen)
        fake = outputs

        g_loss = criterion(fake, real_labels)

        reset_grad()

        g_loss.backward()
        g_optimizer.step()

        if(i%10 == 0):

            a = gen[0]
            a = a.view(28,28)

            if(i % 20 == 0):
                torchvision.utils.save_image(a.detach(), 'test_%d_%d.png'%(epoch, i), normalize = True, range = (0, 255))
            
            print('Discriminator Loss: {}'.format(d_total_loss.item()), end=' ')
            print('Generator Loss: {}'.format(g_loss.item()))

    print('-------------------------epoch-----------------------------------')


        

        

        

