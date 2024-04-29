
from Encoder import *
from Decoder import *
from model import *
import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
import torch.optim as optim 

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 1)
])

mnist = torchvision.datasets.MNIST('./', download=True, transform=my_transform)

input_dim = 28*28
batch_size = 128
num_epochs = 100 
learning_rate = 0.001
hidden_size = 512
latent_size = 8

device = torch.device("mps")
dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

encoder = Encoder(input_dim, hidden_size, latent_size)
decoder = Decoder(latent_size, hidden_size, input_dim)

vae = VAE(encoder, decoder).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for inputs, _ in dataloader:
        inputs = inputs.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        p_x, q_z = vae(inputs)
        log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
        kl = torch.distributions.kl_divergence(
            q_z,
            torch.distributions.Normal(0,1)
        ).sum(-1).mean()
        loss = -(log_likelihood-kl)
        loss.backward()
        optimizer.step()
        
    print(epoch, loss.item(), log_likelihood.item(), kl.item())
        