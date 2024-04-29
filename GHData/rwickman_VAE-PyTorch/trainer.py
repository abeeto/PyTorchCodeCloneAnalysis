from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import MNIST
import torchvision
import matplotlib.pyplot as plt

from model import VAE
from model_util import loss_fn

device = "cuda"

def plot(x_preds, plot_size):
    """Plot resulting images."""
    x_preds = x_preds.detach().cpu() * 0.3081 + 0.1307
            
    fig, axs = plt.subplots(plot_size[0], plot_size[1])
    for  i in range(plot_size[0]):
        for j in range(plot_size[1]):
            axs[i][j].imshow(x_preds[plot_size[1] * i + j][0])

    plt.show()

        
class Trainer:
    def __init__(self, args):
        self.args = args
        self.vae = VAE().to(device)
        self.optim = optim.Adam(self.vae.parameters(), lr=self.args.lr)

        
        train_ds = MNIST('dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        test_ds = MNIST('dataset/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

        self.train_dataloader = DataLoader(
            train_ds,
            self.args.batch_size,
            num_workers = 8,
            prefetch_factor=16,
            drop_last = True,
            shuffle=True)

        self.test_dataloader = DataLoader(
            test_ds,
            self.args.batch_size,
            num_workers = 8,
            prefetch_factor=16,
            drop_last = True,
            shuffle=True)
        
        if self.args.load:
            self.load()

    def load(self):
        model_dict = torch.load("model.pt")
        self.vae.load_state_dict(model_dict["model"])
        self.optim.load_state_dict(model_dict["optimizer"])

    def save(self):
        model_dict = {
            "model": self.vae.state_dict(),
            "optimizer": self.optim.state_dict()
        }

        torch.save(model_dict, "model.pt")

    def train(self):
        for i in tqdm(range(self.args.epochs)):
            for img, label  in self.train_dataloader:
                # Make a model prediction
                img = img.to(device)                
                x_pred, mu, log_var = self.vae(img)

                # Train the VAE
                loss = loss_fn(img, x_pred, mu, log_var)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            self.save()

    
    def test(self, plot_size = (9, 9)):
        self.vae.eval()
        
        # Reconstruction test
        print("PLOTING TEST RECONSTRUCTION")
        for img, label in self.test_dataloader:    
            x_preds, _, _ = self.vae(img.to(device))
            print("x_preds.shape", x_preds.shape)
            plot(x_preds, (4, 8))
            break
        
        # Sample numbers
        print("PLOTTING RANDOM SAMPLES")
        x_preds = self.vae.generate(torch.randn(plot_size[0] * plot_size[1], 256).to(device))
        plot(x_preds, plot_size)

        
