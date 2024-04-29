from torch import nn,optim
import torch.nn.functional as F

class Fashion(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(784,256)
        self.hidden2=nn.Linear(256,64)
        self.output=nn.Linear(64,10)
        
    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
#         x=F.relu(self.output(x))
        x=F.log_softmax(self.output(x), dim=1)
        return x
        
model=Fashion()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)

epochs=5
for i in range(epochs):
    running_loss= 0
    for images, labels in trainloader:
        logits = model(images)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        
    else:
        print(f"Training loss: {running_loss}")





# Plotting the image and probabilities
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))


helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
