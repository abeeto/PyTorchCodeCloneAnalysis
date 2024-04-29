import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline

dataset = MNIST(root='data/', download=True, transform=ToTensor())

image, label = dataset[0]
print('image.shape:', image.shape)
#from chw to hwc
plt.imshow(image.permute(1, 2, 0), cmap='gray')
print('Label:', label)
print(len(dataset))
validation_size=len(dataset)*0.2
training_size=len(dataset)*0.8
train_ds, valid_ds = random_split(dataset,[int(training_size),int(validation_size)])
print(len(train_ds))
print(len(valid_ds))

batch_size = 128

train_loader = DataLoader(train_ds, batch_size,shuffle=True, num_workers=4 )
val_loader = DataLoader(valid_ds, batch_size*2, num_workers=4 )

for images,_ in train_loader:
    print('image shape', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1,2,0)))
    break
    
for images, labels in train_loader:
    print('labels: ', labels)
    inputs = images.reshape(-1, 784)
    print('inputs.shape:', inputs.shape)
    break
    
input_size = inputs.shape[-1]
hidden_size = 32
layer1 = nn.Linear(input_size, hidden_size)
layer1_output = layer1(inputs)
layer1_output_direct = inputs@layer1.weight.t() + layer1.bias
layer1_output_direct.shape

torch.allclose(layer1_output, layer1_output_direct, 1e-3)

relu_outputs = F.relu(layer1_output)

output_size = 10
layer2 = nn.Linear(hidden_size, output_size)
layers2_outputs = layer2(relu_outputs)
print(layers2_outputs.shape)

inputs.shape

F.cross_entropy(layers2_outputs, labels)
outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias
torch.allclose(outputs, layers2_outputs, 1e-3)
outputs2 = (inputs @ layer1.weight.t() + layer1.bias) @ layer2.weight.t() + layer2.bias

combined_layer = nn.Linear(input_size, output_size)

combined_layer.weight.data = layer2.weight @ layer1.weight
combined_layer.bias.data = layer1.bias @ layer2.weight.t() + layer2.bias
outputs3 = inputs @ combined_layer.weight.t() + combined_layer.bias
torch.allclose(outputs2, outputs3, 1e-3)


##Defining model----------------------------------

class MnistModel(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.Linear1 = nn.Linear(in_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        xb = xb.view(xb.size(0), -1) #flatten(Re check)
        out = self.Linear1(xb)
        out = F.relu(out) # act func
        out = self.linear2(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images) #generateing pred
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss':loss, 'val_acc':acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


input_size = 784
hidden_size = 32 
num_classes = 10
    
model = MnistModel(input_size, hidden_size=32, out_size=num_classes)
        
for t in model.parameters():
    print(t.shape)
    
for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)

history = [evaluate(model, val_loader)]
history
history += fit(5, 0.5, model, train_loader, val_loader)
history += fit(5, 0.1, model, train_loader, val_loader)
losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');
accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');

        
