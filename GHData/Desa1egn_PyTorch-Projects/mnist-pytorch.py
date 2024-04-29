'''
EXCERCISE ON CNN NEURAL NET WORK IN PYTORCH FOR THE MNIST DATASET.

CREATED BY:  DES
    
DATE: 10.10.2022

'''
# IMPORT DEPENDENCIES

import torch 
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

# CHECK AVAILABLE DEVICES

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# IMPORT THE DATA (MNIST FROM PYTOTCH DATASETS)

train_set = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
data_loader = DataLoader(train_set, 32)


# BUID THE NEURAL NET WORK

class MNISTClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)


# CREATE INSTANCES OF THE NEWRAL NET WORK
cnn_clf = MNISTClassifier().to(device)

# DEFINE THE OPTIMIZER
opt = Adam(cnn_clf.parameters(), lr=1e-3)

    
# DEFINE THE LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()


# START TRAINING CYCLE
if __name__ == "__main__":
    
    for epoch in range(10):
        for batch in data_loader:
            
            X, y = batch
            X, y = X.to(device), y.to(device)
            
            y_pred = cnn_clf(X)
            loss = loss_fn(y_pred, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            
        print(f"epoch : {epoch} loss is  {loss.item()}")
        
    with open('model_state.pt', 'wb') as f:
        save(cnn_clf.state_dict(), f)
        
    with open('model_state.pt', 'rb') as f:
        cnn_clf.load_state_dict(load(f))        
        
    
    # SAMPLE IMAGES FOR TEST. THE IMAGE BELOW IS A "7" FOR EXAMPLE.    
    image = Image.open('../input/mnist-sample-images/img_12.jpg')
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    print(torch.argmax(cnn_clf(image_tensor)))      
        
            
            



    
