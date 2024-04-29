#libraries

import numpy as np
import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#setting device
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyperparameters
num_epochs=5
batch_size=2
learning_rate=0.001

#transform
transform=transforms.ToTensor()

#download datasets
train_dataset=datasets.CelebA(root="./data",
                         split="train",
                         download=True,
                         transform=transform)
test_dataset=datasets.CelebA(
    root=".data/",
    split="test",
    download=True,
    transform=transform
)

# #loading datasets
train_loader=torch.utils.data.DataLoader("./data/celeba",batch_size=batch_size,shuffle=True)
# test_loader=torch.utils.data.Dataloader(test_dataset,batch_size=batch_size,shuffle=False)
# print("train loader",train_loader.next())
# print("test loader",test_loader)
# ##quick show some samples of datasets
def imshow(img):
    img_np=img.numpy()
    print(img_np.shape)
    plt.imshow(np.transpose(img_np,(1,2,0)))
    plt.show()
dataiter=iter(train_loader)
images,labels=dataiter.next()
imshow(torchvision.utils.make_grid(images))
# #Creating model

#calling class

#loss and optimizer functions

# training loop


#saving model         


# validation

