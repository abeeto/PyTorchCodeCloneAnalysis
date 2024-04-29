from core.dcgan import Discriminator
from core.dcgan import Generator
from core.helpers import info
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from sklearn.utils import shuffle
from imutils import build_montages
from torch.optim import Adam
from torch.nn import BCELoss
from torch import nn
import numpy as np
import argparse
import torch
import cv2
import os

# Custom weight init for generator and discriminator
def weights_init(model):
    # Get the class name
    classname = model.__class__.__name__
    
    if classname.find("Conv") != -1:
        # Init the weight from normal distributions
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        
    elif classname.find("Batch") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
        
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="output/", help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=20, help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128, help="batch size for training")
args = vars(ap.parse_args())

NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)]
    )

info("Loading MNIST dataset.")
train_data = MNIST(root="data", train=True, download=True, transform=data_transforms)
test_data = MNIST(root="data", train=False, download=True, transform=data_transforms)
data = torch.utils.data.ConcatDataset((train_data, test_data))

data_loader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
steps_per_epoch = len(data_loader.dataset) // BATCH_SIZE

info("Building Generator.")
gen = Generator(100, 512, 1)
gen.apply(weights_init)
gen.to(DEVICE)

info("Building Discriminator.")
dis = Discriminator(depth=1)
dis.apply(weights_init)
dis.to(DEVICE)

gen_opt = Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002/NUM_EPOCHS)
dis_opt = Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002/NUM_EPOCHS)

criterion = BCELoss()

info("Starting Training.")
benchmark_noise = torch.randn(256, 100, 1, 1, device=DEVICE)

real_label = 1
fake_label = 0

for epoch in range(NUM_EPOCHS):
    info("Starting Epoch {} of {}".format(epoch+1, NUM_EPOCHS))
    
    epoch_loss_g = 0
    epoch_loss_d = 0
    
    for x in data_loader:
        # Zero out the Discriminator gradient
        dis.zero_grad()
        
        images = x[0]
        images = images.to(DEVICE)
        
        # Get the batch size
        bs = images.size(0)

        # Create a labels tensor
        labels = torch.full((bs,), real_label, dtype=torch.float, device=DEVICE)
        
        # Forward pass through discriminator
        output = dis(images).view(-1)

        # Calculate the loss on all-real labels
        error_real = criterion(output, labels)
        
        # Calculate gradients by performing a backward pass
        error_real.backward()
        
        # Randomly generate noise for the generator to predict on
        noise = torch.randn(bs, 100, 1, 1, device=DEVICE)
        
        # Generate a fake image batch
        fake = gen(noise)
        
        # Replace the value of labels
        labels.fill_(fake_label)
        
        # Forward pass through discriminator using fake data
        output = dis(fake.detach()).view(-1)
        error_fake = criterion(output, labels)
        
        # Calculate gradients by performing a backward pass
        error_fake.backward()
        
        error_d = error_real + error_fake
        dis_opt.step()
        
        # Zero out the Generator gradient
        gen.zero_grad()
        
        labels.fill_(real_label)
        output = dis(fake).view(-1)
        
        error_g = criterion(output, labels)
        error_g.backward()
        
        gen_opt.step()
        
        epoch_loss_d += error_d
        epoch_loss_g += error_g
        
    info("Generator Loss: {:4f}, Discriminator Loss: {:4f}".format(epoch_loss_g / steps_per_epoch, epoch_loss_d / steps_per_epoch))

    if (epoch + 1) % 2 == 0:
        # Set the generator to evaluation mode
        gen.eval()
        
        images = gen(benchmark_noise)
        images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
        
        # Scale images back to the [0, 255] range
        images = ((images * 127.5) + 127.5).astype("uint8")
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (16, 16))[0]
        
        p = os.path.join(args["output"], "epoch_{}.png".format(str(epoch+1).zfill(4)))
        cv2.imwrite(p, vis)
        
        # Set the generator to training mode
        gen.train()
