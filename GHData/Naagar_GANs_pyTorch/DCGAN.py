# DCGAN 

import torch 
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_utils import Discriminator, Generator 


# Hyperparameters 

lr = 0.0002
batch_size = 256
image_size = 64  #  28 x 28 --> 64 x 64
channels_img = 1
channels_noise = 256
num_epochs = 110

features_d = 16 
features_g = 16

my_transforms = transforms.Compose([

	transforms.Resize(image_size),
	transforms.ToTensor(),
	transforms.Normalize((0.5), (0.5), (0.5))
	]) 
dataset = datasets.MNIST(root='dataset/', train=True, transform=my_transforms, download=True)

dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True )

#set device 
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# Create Discriminator and Generator 

netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)


# SetUp optimizer for G and D

optimizerD = optim.Adam(netD.parameters(), lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr, betas=(0.5, 0.999))


netG.train()
netD.train()

criterion = nn.BCELoss()


real_label = 1
fake_label =0


writer_real = SummaryWriter(f'runs/GAN_MNIST/test_real')
writer_fake = SummaryWriter(f'runs/GAN_MNIST/test_fake')



fixed_nise = torch.randn(64, channels_noise, 1, 1).to(device)

print('Starting teaning: ')

for epoch in range(num_epochs):
	for batch_idx, (data, targets) in enumerate(dataloader):
		data = data.to(device)
		batch_size = data.shape[0]

		## train Discriminator: max log(D(x)) + log(1- D(G(z)))
		netD.zero_grad()
		label = (torch.ones(batch_size)*0.9).to(device)  # a hack to 0.9
		output = netD(data).reshape(-1)
		# print(output.shape, label.shape)
		lossD_real = criterion(output, label)
		D_x = output.mean().item()

		noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
		fake = netG(noise)

		label = (torch.ones(batch_size)*0.1).to(device)   # a hack 0.1

		output = netD(fake.detach()).reshape(-1)
		lossD_fake = criterion(output, label)

		lossD = lossD_real + lossD_fake 
		lossD.backward()

		optimizerD.step()

		## train the Generator: max log(D(G(z)))

		netG.zero_grad()
		label = torch.ones(batch_size).to(device)
		output = netD(fake).reshape(-1)
		lossG = criterion(output, label)

		lossG.backward()
		optimizerG.step()

		if batch_idx %500 ==0 :
			print(f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} Loss D: {lossD:.4f}, Loss G: {lossG:.4f} D(x) : {D_x:.4f}')

			with torch.no_grad():
				fake = netG(fixed_nise)

				img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
				img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
				writer_real.add_image("Mnist Real Images", img_grid_real)
				writer_real.add_image("Mnist Fake Images", img_grid_fake)
				
				