# Autoencoder first try from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

import torch
# Data Loading and normalizing using torchvision
import torchvision
import torchvision.transforms as transforms
# use torch.nn for neural networks and torch.nn.functional for functions!
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Path for Data
data_path = './data'
# Path to save and load model
net_path = './models/AE_net.pth'

# set up the divice (GPU or CPU) via input prompt
cuda_true = input("Use GPU? (y) or (n)?")
if cuda_true == "y":
    device = "cuda"
else:
    device = "cpu"
print("Device:", device)

# Hyperparameters
num_epochs = 40
train_batch_size = 64
test_batch_size = 10
learning_rate = 0.001

# imports to show pictures
import matplotlib.pyplot as plt

# save a networks parameters for future use
def save_network(net: nn.Module, path):
    # save the network?
    save = input("Save net? (y) or (n)?")
    if save == "y":
        torch.save(net.state_dict(), path)
    else:
        pass


# load an existing network's parameters and safe them into the just created net
def load_network(net: nn.Module, net_path):
    # save the network?
    load = input("Load Network? (y) or (n)?")
    if load == "y":
        net.load_state_dict(torch.load(net_path))
    else:
        pass

# Dataset loading and transforming

transform = transforms.Compose([
    transforms.ToTensor()
])


if __name__ == "__main__":

    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root=data_path, train=False, download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=0)

    # Autoencoder

    class AE(nn.Module):
        def __init__(self, **kwargs):
            super(AE, self).__init__()
            self.encoder_hidden_layer = nn.Linear(
                in_features=kwargs["input_shape"], out_features=128
            )
            self.encoder_output_layer = nn.Linear(
                in_features=128, out_features=128
            )
            self.decoder_hidden_layer = nn.Linear(
                in_features=128, out_features=128
            )
            self.decoder_output_layer = nn.Linear(
                in_features=128, out_features=kwargs["input_shape"]
            )

        def forward(self, x):
            activation = self.encoder_hidden_layer(x)
            activation = torch.relu(activation)
            code = self.encoder_output_layer(activation)
            code = torch.sigmoid(code)
            activation = self.decoder_hidden_layer(code)
            activation = torch.relu(activation)
            activation = self.decoder_output_layer(activation)
            reconstructed = torch.sigmoid(activation)
            return reconstructed


    # instantiate the AE
    model = AE(input_shape=28*28)
    load_network(model, net_path)
    model = model.to(device=device)


    # optimizer amd loss -> Adam and Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # train loop
    def train(nEpochs: int):
        for epoch in range(nEpochs):
            loss = 0
            for data in trainloader:
                batch_features, _ = data
                # reshape mini-batch data to [N, 28*28] matrix
                # load it to the active device
                batch_features = batch_features.view(-1, 28 * 28).to(device=device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                train_loss = criterion(outputs, batch_features)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()

            loss = loss / len(trainloader)

            print("epoch : {}/{}, recon_loss = {:.8f}".format(epoch + 1, nEpochs, loss))


    def check_reconstruction(loader: DataLoader, model):
        model.eval()
        test_examples = None
        with torch.no_grad():
            for batch_features, _ in loader:
                batch_features = batch_features.to(device=device)
                test_examples = batch_features.view(-1, 28 * 28)
                reconstruction = model(test_examples)
                break

        with torch.no_grad():
            plt.figure(figsize=(15, 4))
            for index in range(test_batch_size):
                # display original
                ax = plt.subplot(2, test_batch_size, index + 1)
                plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                #display recon
                ax = plt.subplot(2, test_batch_size, index + 1 + test_batch_size)
                plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

    # execute training!
    train_true = input("Train network? (y) or (n)?")
    if train_true == "y":
        train(num_epochs)

        # save the network?
        save = input("Save net? (y) or (n)?")
        if save == "y":
            torch.save(model.state_dict(), net_path)
        else:
            pass

    else:
        pass

    check_reconstruction(testloader, model)