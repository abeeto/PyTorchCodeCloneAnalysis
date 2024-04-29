import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to the first hidden layer linear transformation
        self.hidden_1 = nn.Linear(784, 128)
        # Inputs to the second hidden layer linear transformation
        self.hidden_2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)

        # Define sigmoid activation function, softmax, and cross-entropy loss
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.logsoftmax(x)

        return x


 # Get the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,), inplace=False)])
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter.next()
inputs = images.view(images.shape[0], -1)

# Define and run network
model = Network()

# Forward pass
logps = model.forward(inputs)

# Define the loss
criterion = nn.NLLLoss()

# Calculathe the loss
loss = criterion(logps, labels)
print(loss)

# # Check
# print('Before backward pass: \n', model.hidden_1.weight.grad)
# output.backward()
# print('After backward pass: \n', model.hidden_1.weight.grad)

# Optimizer require parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

########################################################################################################################
########################################################################################################################
# One iteration of training
# STEP-1 : Make a forward pass through the network
# STEP-2 : Use the network to calculate the loss
# STEP-3 : Perform a backward pass (loss.backward()) to calculate the gradients
# STEP-4 : Take a step with the optimizer to update the weights

print('Initial weights - ', model.hidden_1.weight)

# Clear the gradients, do this becuase gradients are accumulated
optimizer.zero_grad()
# Forward pass, then backward pass, then update weights
output = model.forward(inputs)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model.hidden_1.weight.grad)
# Take a step to update and see the new weights
optimizer.step()
print('Updated weights - ', model.hidden_1.weight.grad)


