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

        # Dropout to prevent overfitting. Setting dropout to 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Making sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # Output (This does not have dropout)
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


# ---------- Training Model ----------
model = Classifier()
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

            # set model to evaluation mode. This disables dropout.
            model.eval()

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

        # set model back to train mode
        model.train()

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

Note:
Compare graph to inference-and-validation.py to see the effect of dropout.

"""
