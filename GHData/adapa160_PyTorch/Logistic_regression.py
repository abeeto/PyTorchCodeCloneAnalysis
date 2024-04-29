import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

%matplotlib inline
from torchvision.datasets import MNIST
dataset = MNIST(root='/data', download = True, train=True, transform=transforms.ToTensor())
img_tensor, label= dataset[0]
print(img_tensor.shape)
train_ds, valid_ds = random_split(dataset,[40000,20000] )
print(len(train_ds))
print(len(valid_ds))

batch_size= 128
train_loader=DataLoader(train_ds, batch_size, shuffle=True)
val_loader=DataLoader(valid_ds, batch_size)
#no of pixels
#no of outputs
input_size= 28*28
output_size= 10

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # Epoch history save
    
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
    
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))
    #Acc returns true for uneq element and false for equal divided by no of image for accuracy

    
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    

        
        
    def forward(self, x):
        x = x.reshape(-1, 784)
        out = self.linear(x)
        return out
    
    def training_step(self, batch):
        img, labels = batch
        out = self(img)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def validation_step(self, batch):
        img, labels = batch
        out = self(img)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
        
    
model = MnistModel()

result0 = evaluate(model, val_loader)
result0

history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)
history3 = fit(5, 0.001, model, train_loader, val_loader)
history4 = fit(5, 0.001, model, train_loader, val_loader)
history5 = fit(5, 0.001, model, train_loader, val_loader)

history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');

#for images, labels in train_loader:
#    print(labels)
#    print(images.shape)
 #   output= model.forward(images)
#    print(output)
 #   break




#for i in range(0, 7):
    #img, labels = dataset[i]
    #plt.imshow(img, cmap='gray')
    #print('Labels:',labels)

    
    
    
#-----------------Checking with indi image
def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

