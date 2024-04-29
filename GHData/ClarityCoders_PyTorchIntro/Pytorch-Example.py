import torch
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # image 28 X 28 = 784
        self.input_layer = nn.Linear(784, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 10)
    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        data = F.relu(self.hidden2(data))
        data = self.output(data)
        
        return F.log_softmax(data, dim=1)
        

training = datasets.MNIST("", train=True, download=True, 
                          transform = transforms.Compose([transforms.ToTensor()]))

testing = datasets.MNIST("", train=False, download=True, 
                          transform = transforms.Compose([transforms.ToTensor()]))


train_set = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)

network = Network()

learn_rate = optim.Adam(network.parameters(), lr=0.001)
epochs = 4

for i in range(epochs):
    for data in train_set:
        image, output = data
        network.zero_grad()
        result = network(image.view(-1,784))
        loss = F.nll_loss(result,output)
        loss.backward()
        learn_rate.step()
    print(loss)


# Test the network
network.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in test_set:
        image, output = data
        result = network(image.view(-1,784))
        for index, tensor_value in enumerate(result):
            total += 1
            if torch.argmax(tensor_value) == output[index]:
                correct += 1
                
accuracy = correct / total
print(f"Accuracy: {accuracy}")


# Look image processing
from PIL import Image
import numpy as np
import PIL.ImageOps   

img = Image.open("firstTest.png")
img = img.resize((28,28))
img = img.convert("L")
img = PIL.ImageOps.invert(img)

plt.imshow(img)

img = np.array(img)
img = img / 255
image = torch.from_numpy(img)
image = image.float()

result = network.forward(image.view(-1,28*28))
print(torch.argmax(output))
