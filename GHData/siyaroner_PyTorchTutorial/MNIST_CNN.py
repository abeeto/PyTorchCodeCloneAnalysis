import numpy as np
import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear,Conv2d,ReLU,MaxPool2d
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyper parameters
num_epochs=5
batch_size=4
learning_rate=0.001
#transform
transform=transforms.ToTensor()

# downloading MNIST dataset
train_dataset=datasets.MNIST(root="./data",
                         train=True,
                         download=True, 
                         transform=transform)
test_dataset=datasets.MNIST(root="./data",
                         train=False,
                         download=True, 
                         transform=transform)
# loading MNIST dataset
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
##quick show some samples of datasets
# def imshow(img):
#     # img=img/2+0.5 #unnormalize
#     img_np=img.numpy()
#     print(img_np.shape)
#     plt.imshow(np.transpose(img_np,(1,2,0)))
#     plt.show()
    
    
# # get some random trainning images
# dataiter=iter(train_loader)
# images,labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
## creating CNN

#Creating model
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1=Conv2d(1,6,5)
        self.pool=MaxPool2d(2,2)
        self.conv2=Conv2d(6,24,5)
        self.fc1=Linear(24*4*4,120)
        self.fc2=Linear(120,84)
        self.fc3=Linear(84,10)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        # print('x_shape:',x.shape)
        x=x.view(4,24*4*4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

#calling class
model=MNIST_CNN().to(device)
#loss and optimizer functions
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
n_total_steps=len(train_loader)
# training loop
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        
        #forward pass
        outputs=model(images)
        # print(outputs.shape,labels.shape)
        loss=criterion(outputs,labels)
        
        #backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%5000==0:
          print(f"Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")  

#saving model         
print("Training is done")         
path="./mnist_cnn.pth"
torch.save(model.state_dict(),path)

# validation
with torch.no_grad():
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in range(10)]
    n_class_samples=[0 for i in range(10)]
    
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        #max returns (max,index)
        _,prediction=torch.max(outputs,1)
        n_samples +=labels.size(0)
        n_correct +=(prediction==labels).sum().item()
        
        for i in range(batch_size):
            label=labels[i]
            pred=prediction[i]
            if (label==pred):
                n_class_correct[label] +=1
            n_class_samples[label] +=1
    acc=100.0*n_correct/ n_samples
    print(f"Accuracy of the network: {acc} %")
    
    for i in range(10):
        acc=100.0*n_class_correct[i]/n_class_samples[i]
        print(f"Accuracy of {classes[i]} : {acc} %")
      
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          

    