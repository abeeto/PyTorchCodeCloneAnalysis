'''
Breakdown:

    1. Importation of modules
    2. Device configuration and hyper-parameters
    3. Dataloader, Transformation
    4. Creation of Logistic regression model
    5. Loss anf Optimizer
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

'''Device configuration'''
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')

'''Hyper Parameters'''
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

'''Dataloader and Transformations (On MNIST Dataset)'''
#MNIST Dataset (images and labels)
training_ds = dataset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
testing_ds = dataset.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#dataloader (input pipeline)
train_loader = DataLoader(dataset=training_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testing_ds, batch_size=batch_size, shuffle=False)

'''Creation of Logistic regression model'''
model = nn.Linear(input_size, num_classes)

'''Loss and Optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''Training Loop'''
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        #reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

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
               