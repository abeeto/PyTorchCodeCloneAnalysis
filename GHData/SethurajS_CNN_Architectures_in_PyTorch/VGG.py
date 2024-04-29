# Importing the requirements
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Building the VGG16 model
VGG_16 = [64, 64, 'MAX', 128, 128, 'MAX', 256, 256, 256, 'MAX', 512, 512, 512, 'MAX', 512, 512, 512, 'MAX']
VGG_19 = [64, 64, 'MAX', 128, 128, 'MAX', 256, 256, 256, 256, 'MAX', 512, 512, 512, 512, 'MAX', 512, 512, 512, 512, 'MAX']

class VGG(nn.Module):
    def __init__(self, in_channels=3, output_samples=10, VGG_arc=None):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.output_samples = output_samples
        self.conv_layers = self.create_conv_layer(VGG_arc)
        self.fcl = nn.Sequential(
            nn.Linear(512*1*1, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.output_samples)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcl(x)
        return x

    def create_conv_layer(self, VGG16_channels):
        layers = []
        for channel in VGG16_channels:
            if type(channel) == int :
                out_channels = channel
                layers += [nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(channel),
                           nn.ReLU()]
                self.in_channels = channel

            elif channel == 'MAX':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)  

# Device
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# Defining Hyperparameters
learning_rate = 1e-3
batch_size = 64
num_epochs = 10
input_channels = 3
output_classes = 10
load_model=False

# Calling Model 
model = VGG(in_channels=input_channels, output_samples=output_classes, VGG_arc=VGG_19).to(device=device)

# Getting the datasets
train_data = datasets.CIFAR10(root="cifar10_datasets/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.CIFAR10(root="cifar10_datasets/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Setting loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Saving model checkpoint
def save_checkpoint(checkpoint, filename):
    print("\nSaving checkpoint...\n")
    torch.save(checkpoint, filename)

# Loading model checkpoint
def load_checkpoint(checkpoint):
    print("\nLoading checkpoint...\n")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optmizer'])

if load_model:
    load_checkpoint(torch.load("model_checkpoint.pth.tar"))

print("\n")
print(">>>> Training !!")
print("\n")
# Training 
for epoch in range(1, num_epochs+1):
    losses = []
    if epoch % 2 == 0:
        checkpoint = {"state_dict" : model.state_dict(), "optmizer" : optimizer.state_dict()}
        save_checkpoint(checkpoint, "VGG_model_checkpoint/model_checkpoint.pth.tar")
    for data_indx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    mean_loss = sum(losses)/len(losses)
    print("Epoch : {} -- Loss : {}".format(epoch, mean_loss))


def evaluate_model(dataset=datasets, model=model):
    if dataset.dataset.train:
        print(">>>> Evaluating accuracy on train dataset :")
    else:
        print("\n")
        print(">>>> Evaluating accurcy on test dataset : ")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataset:
            X = X.to(device=device)
            y = y.to(device=device)

            prediction = model(X)
            _, prediction = prediction.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        print("     Got {}/{} with accuracy of {}% !".format(num_correct, num_samples, (float(num_correct)/float(num_samples))))

    model.train()


evaluate_model(dataset=train_loader, model=model)
evaluate_model(dataset=test_loader, model=model)