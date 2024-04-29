import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# Model
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
    
model = Classifier()

images, labels = next(iter(testloader))
# Get the class probabilities
ps = torch.exp(model(images))
# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
print(ps.shape)


# ---------- Pre training accuracy ----------
# With the probabilities, we can get the most likely class using the `ps.topk` method. This returns the k highest
# values. Since we just want the most likely class, we can use `ps.topk(1)`. This returns a tuple of the top-$k$
# values and the top-$k$ indices. If the highest value is the fifth element, we'll get back 4 as the index.
top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 10 examples
print(top_class[:10,:])


# Now we can check if the predicted classes match the labels. This is simple to do by equating top_class and labels,
# but we have to be careful of the shapes. Here top_class is a 2D tensor with shape (64, 1) while labels is 1D with
# shape (64). To get the equality to work out the way we want, top_class and labels must have the same shape.
equals = top_class == labels.view(*top_class.shape)


# Now we need to calculate the percentage of correct predictions. equals has binary values, either 0 or 1. This means
# that if we just sum up all the values and divide by the number of values, we get the percentage of correct
# predictions. This is the same operation as taking the mean, so we can get the accuracy with a call to torch.mean

# equals has type torch.ByteTensor but torch.mean isn't implemented for tensors with that type. So we'll need to
# convert equals to a float tensor. Note that when we take torch.mean it returns a scalar tensor, to get the actual
# value as a float we'll need to do accuracy.item()
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')


# ---------- Training Model ----------
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        # Clear the gradients
        optimizer.zero_grad()

        # Calculate Output
        log_ps = model(images)

        # Calculate Loss
        loss = criterion(log_ps, labels)

        # Calculate gradient using autograd
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    else:
        ## TODO: Implement the validation pass and print out the validation accuracy
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for images, labels in testloader:

                # Calculate output
                log_ps = model(images)

                # Calculate Loss
                test_loss += criterion(log_ps, labels)

                # Output as probabilities instead of log probabilities
                probabilitiesOutput = torch.exp(log_ps)

                # Findng the class with the top probability
                top_p, top_class = probabilitiesOutput.topk(1, dim=1)

                # Calculating matches between model output and labels and storing in equals. Note shapes are not same.
                equals = top_class == labels.view(*top_class.shape)

                # Calculating accuracy by taking % of times we get the right prediction. In this case it equals mean.
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

"""
Notes:
The graph clearly shows the impact of overfitting. The model continues to improve on the trianing set, but after an 
initial drop in loss on test data the loss starts increasing. We solve for this issue using drop out. 

"""

