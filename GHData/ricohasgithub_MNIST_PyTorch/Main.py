import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import numpy as np

from CNN import CNN

train_dataset=dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset=dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

model=CNN(out_1=16,out_2=32)

criterion=nn.CrossEntropyLoss()

learning_rate=0.1
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100)
validation_loader=torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=5000)

n_epochs=10
loss_list=[]
accuracy_list=[]
N_test=len(validation_dataset)
#n_epochs
for epoch in range(n_epochs):
    for x, y in train_loader:
      

        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z=model(x)
        # calculate loss 
        loss=criterion(z,y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()
        
        
        
    correct=0
    #perform a prediction on the validation  data  
    for x_test, y_test in validation_loader:

        z=model(x_test)
        _,yhat=torch.max(z.data,1)

        correct+=(yhat==y_test).sum().item()
        
   
    accuracy=correct/N_test

    accuracy_list.append(accuracy)
    
    loss_list.append(loss.data)
