import os
import torch
from torch import nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as Transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a sample result directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper Parameters
IMAGE_SIZE = 784
# H_DIM = 400
NUM_EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False

# Mnist digits dataset

data_transforms = Transforms.Compose([
    # Converts a PIL.Image or numpy.ndarray to torch.Tensor of shape (C x H x W) and normalize in the range [-1.0, 1.0]
    Transforms.ToTensor(),
    Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
])
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform= data_transforms,    
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (64, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_SIZE, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 10),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, IMAGE_SIZE),
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-5)

# Print model's state_dict
print("Auto-Encoder's state_dict:")
for param_tensor in autoencoder.state_dict():
    print(param_tensor, "\t", autoencoder.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

for epoch in range(NUM_EPOCH):
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device).view(-1, IMAGE_SIZE) # shape (batch, 28*28)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print("Epoch[{}/{}], Step[{}/{}], Train Loss: {:.4f}"
                    .format(epoch+1, NUM_EPOCH, i+1, len(train_loader), loss.data.numpy()))
    with torch.no_grad():
        # Save the sampled images and reconstructed images
        img_batch, _ = next(iter(train_loader))
        sample_imgs = img_batch[:5].to(device)
        save_image(sample_imgs, os.path.join(sample_dir, 'sample-{}.png'.format(epoch+1)))
        _, out = autoencoder(sample_imgs.view(-1, IMAGE_SIZE))
        save_image(out.view(-1, 1, 28, 28), os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

# save model
# create a model directory if not exists
trained_model = './trained_model'
if not os.path.exists(trained_model):
    os.makedirs(trained_model)
torch.save(autoencoder, trained_model + '/autoencoder.pth')
# # load model
# # Model class must be defined somewhere
# model = torch.load(PATH)
# model.eval()