# feed forward Neural Network
import torch
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F # calls partent as a function instead of the module?

import torch.optim as optim
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # image 28 x 28 = 784 px
        self.input_layer = nn.Linear(784, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 10) # numbers 0-9
    def forward(self, data):
        data = F.relu(self.input_layer(data)) # rectified linear unit, to see what nodes are firing/not firing
        data = F.relu(self.hidden1(data))
        data = F.relu(self.hidden2(data))
        data = self.output(data)

        return F.log_softmax(data, dim=1) # dim=1 dimension of our output data, single set of choices 0-9
        # softmax gives you probability for event or class

training = datasets.MNIST("", train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

testing = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)

network = Network()

learn_rate = optim.Adam(network.parameters(), lr=0.01) # how much to adjust weights to make a better prediction
# too high or too low not good
# first var is the network's parameters. Which in this case is just inherited from parent module
epochs = 4

# training needs a for loop
for i in range(epochs):
    for data in train_set:
       image, output = data 
       network.zero_grad() # so each image is unique
       result = network(image.view(-1, 784))
       loss = F.nll_loss(result, output)
       loss.backward()
       learn_rate.step()
    print(loss)
    #    print('-'*20)
    #    print(image)
    #    plt.imshow(image[0].view(28,28))
    #    print('-'*20)
    #    print(output)
    #    print('-'*20)
    #    break
# put out network on eval mode, eval the test data not training anymore

network.eval()
correct = 0
total = 0 # counter of correct images

with torch.no_grad():
    # no epochs needed anymore
    for data in test_set:
        image, output = data # unpacking the tuple again
        result = network(image.view(-1, 784)) 
        for index, tensor_value in enumerate(result):
            total += 1
            if torch.argmax(tensor_value) == output[index]: # find max value in tensor object, highest probability is the guess
                correct += 1

accuracy = correct/total
print(f"Accuracy: {accuracy}")

# Look at image processing
from PIL import Image
import numpy as np
import PIL.ImageOps

img = Image.open("firstTest.png")
img = img.resize((28,28)) # need to resize the image before greyscale
img.convert("L") # convert to grey scale, taken from PIL lib
img = PIL.ImageOps.invert(img)

plt.imshow(img)

# turn image into numpy arr and then tensor torch to use in NN
img = np.array(img) 
img = img / 255 # to normalize the data (make between 0 and 1)
image = torch.from_numpy(img) 
image = image.float() #torch needs a float

result = network.forward(image.view(-1, 28*28))
print(torch.argmax(output))

