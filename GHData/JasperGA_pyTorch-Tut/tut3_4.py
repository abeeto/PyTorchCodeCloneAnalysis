import torch 
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.Module):

    def __init__(self):
        super().__init__() # need this to copy initizaltion of parent class
        # defining each layer in the neural network
        self.fc1 = nn.Linear(28*28, 64) # nn.Linear(input, output) input is the flatten image (28*28 pixels) output is whatever we want
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) # final layer outputs 10 classes, its deciding which number from 0-9

    def forward(self, x):
    	x = F.relu(self.fc1(x)) # use rectified linear activation function
    	x = F.relu(self.fc2(x))
    	x = F.relu(self.fc3(x))
    	x = self.fc4(x)
    	return F.log_softmax(x, dim=1) # make the output layer sums to one
    	# can add logic to above to make it more fancy, like have different layers for different situations

net = Net()
print(net)

X = torch.rand((28,28)) #simulate an image
X = X.view(-1, 28*28) # need to flatten image first
output = net(X)

print(output)



# Tut 4
import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3
for epoch in range(EPOCHS):
	for data in trainset:
		# data is a batch of featuresets and labels
		X,y = data
		net.zero_grad() # start at zero for each batch
		output = net(X.view(-1, 28*28))
		loss = F.nll_loss(output, y) # use different loss functions depending on what the output is
								# for this its just scaler (e.g 4)
		loss.backward()
		optimizer.step()

	print(loss)

# show the accuracy of the model
correct = 0
total = 0

with torch.no_grad(): # use no_grad cause we dont want the gradient to be changed cause that would be
					# optimizing, we just wanna know how the current model is
	for data in trainset:
		X, y = data
		output = net(X.view(-1, 784))
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Accuracy: ", round(correct/total, 3))

import matplotlib.pyplot as plt
plt.imshow(X[3].view(28,28))
plt.show()

print(torch.argmax(net(X[3].view(-1,784))[0])) 