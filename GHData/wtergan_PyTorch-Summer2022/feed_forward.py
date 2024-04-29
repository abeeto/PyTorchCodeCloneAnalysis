'''Breakdown:

    1. Importation of modules
    2. Device configuration and hyper-parameters
    3. Dataloader, Transformation
    4. Creation of Neural Network
    5. Loss and Optimizer
    6. Training Loop (batch training)
    7. Model evaluation

    (This example utilizes the MNIST Dataset)

'''

'''Importation of modules'''
import torch, torchvision
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''Device configuration'''
#GPU will be slower than cpu in this case since not woeking on large dataset
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')

'''Hyper Parameters'''
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

'''Dataloader and Transformations (On MNIST Dataset)'''
training_ds = dataset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
testing_ds = dataset.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=training_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testing_ds, batch_size=batch_size, shuffle=False)

'''Samples of the MNIST'''
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray') 
plt.show()    

'''Creation of Neural Network (use of .Sequential for easier implementation'''
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.reLu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.reLu(out)
        out = self.linear2(out)
        return out    

model = NN(input_size, hidden_size, num_classes)
print(f'Model:\n\n{model}\n')

'''Loss and Optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   

'''Training Loop'''
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #same as flattening the images. move to configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, n_total_steps, loss.item()))

'''Model Evaluation (Testing)'''
#note that we do not need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100*correct/total} %')        

#save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')    