import torch
from torch import nn

#inheriting from nn.Module
# Example. not using this
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear
        # Creates a module for a linear transformation, 𝑥𝐖+𝑏xW+b, with 784 inputs and 256 outputs
        self.hidden = nn.Linear(784, 256)
        # Output Layer, 10 units - for each digit
        self.output = nn.Linear(256,10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        #pass the input tensor through all the operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

#model = Network()
#model


# Prepare data for Training

import torch.nn.functional as F
from torchvision import datasets, transforms
# Transform to normalize the data
# Converts the image into numbers, that are understandable by the system. It separates the image into three color channels (separate images):
# red, green & blue. Then it converts the pixels of each image to the brightness of their color between 0 and 255.
# These values are then scaled down to a range between 0 and 1
# Normalizes the tensor with a mean and standard deviation which goes as the two parameters respectively
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
# Training Data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
valset = datasets.MNIST('~/.pytorch/MNIST_data_test/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# See Data
dataiter = iter(trainloader)
images, labels = dataiter.next()
print("Training Data: ")
print(images.shape)
print(labels.shape)

import matplotlib.pyplot as plt
#display image
figure = plt.figure()
num_of_images = 60
for index in range(1,num_of_images+1):
    plt.subplot(6,10,index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r');
plt.show()


# Neural Network
model = nn.Sequential(nn.Linear(784,128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax(dim=1))
print("Model: ")
print(model)

# Loss
criterion = nn.NLLLoss()

# optimizer and learning rate

optimizer = torch.optim.SGD(model.parameters(), lr = 0.003)

epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # flatten MNIST images in a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training Pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        #print('Loss Before backward pass: \n', model[0].weight.grad)
        loss.backward()
        #print('Loss After backward pass: \n', model[0].weight.grad)
        optimizer.step()

        running_loss +=loss.item()
    else:
        print(f"Training Loss: { running_loss/len(trainloader)}")


# Test Model

images,labels = next(iter(valloader))

img = images[0].view(1,784)
print("\nPredicting the digit in [0][1] position: ")
with torch.no_grad():
    logps=model(img)

ps=torch.exp(logps)
probab=list(ps.numpy()[0])
print("Predicted Digit = ", probab.index(max(probab)))
#view_classify(img.view(1,28,28),ps)

# Counting the correct Prediction

correct_count,all_count =0,0
for images,labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1,784)
        with torch.no_grad():
            logps = model(img)

        ps=torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label=probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count += 1;
        all_count += 1
print("Number of Images tested: ",all_count)
print("\nModel Accuracy= ",(correct_count/all_count))

# save model

torch.save(model, './my_NN_mnist_model.pt')