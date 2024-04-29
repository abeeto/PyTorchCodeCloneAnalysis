from pickletools import optimize
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import transformers
from torch.utils.data import DataLoader, Dataset
import torchvision 
import os
import glob
import pandas as pd
import cv2
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
torch.manual_seed(6)

class_names_label = {'buildings': 0,
                    'forest' : 1,
                    'glacier' : 2,
                    'mountain' : 3,
                    'sea' : 4,
                    'street' : 5
                    }
nb_classes = 6

path_train='C:/Users/nqh/Desktop/im_clas/venv/torch-image-classification/seg_train/seg_train'
path_pred='C:/Users/nqh/Desktop/im_clas/venv/torch-image-classification/seg_pred/seg_pred'
path_test='C:/Users/nqh/Desktop/im_clas/venv/torch-image-classification/seg_test/seg_test'
list_sub=os.listdir(path_train)
# create csv file which contain the class of image

id_image=[]
clas=[]
location=[]
for sub in list_sub:
    list_ima_class=os.listdir(path_train+'/'+sub)
    for ima in list_ima_class:
        id_image.append(ima)
        clas.append(sub)
        location.append(path_train+'/'+sub+'/'+ima)
df=pd.DataFrame({'image':id_image, 'class':clas, 'location':location})
df['class']=df['class'].replace(class_names_label)
df.to_csv('data_train.csv')


id_image=[]
clas=[]
location=[]
for sub in list_sub:
    list_ima_class=os.listdir(path_test+'/'+sub)
    for ima in list_ima_class:
        id_image.append(ima)
        clas.append(sub)
        location.append(path_test+'/'+sub+'/'+ima)
df2=pd.DataFrame({'image':id_image, 'class':clas, 'location':location})
df2['class']=df2['class'].replace(class_names_label)
df2.to_csv('data_test.csv')


class Dataset(Dataset):
    def __init__(self, file_csv):
        super(Dataset, self).__init__()
        self.df=pd.read_csv(file_csv)
        self.name=self.df['image'].values
        self.clas=self.df['class'].values
        self.location=self.df['location'].values
        self.len=len(self.clas)
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image=cv2.imread(self.location[index])
        image=cv2.resize(image,(150,150))
        image = np.moveaxis(image, -1, 0)
        y=self.clas[index]
        return image, y


dataset=Dataset('data_train.csv')
vali_dataset=Dataset('data_test.csv')
# plt.imshow(dataset[3][0])
# print(dataset[3][0].shape)
# plt.show()
# print(dataset)

# Tao model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # l1
        self.cv1=nn.Conv2d(in_channels=3,out_channels=8, kernel_size=3, padding='same')
        self.a1=torch.nn.ReLU()
        # l2
        self.cv2=nn.Conv2d(in_channels=8,out_channels=16, kernel_size=3, padding='same')
        self.a2=torch.nn.ReLU()
        # l3
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        # l4
        self.cv3=nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3, padding='same')
        self.a3=torch.nn.ReLU()
        
        # l5
        self.cv4=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, padding='same')
        self.a4=torch.nn.ReLU()
        # l6
        self.maxpool2=nn.MaxPool2d(kernel_size=3)
        # l7
        self.cv5=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding='same')
        self.a5=torch.nn.ReLU()
        # l8
        self.cv6=nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, padding='same')
        self.a6=torch.nn.ReLU()
        # l9
        self.maxpool3=nn.MaxPool2d(kernel_size=5)
        # l10
        self.flat1=nn.Flatten()

        # l11
        self.fc1=nn.LazyLinear(out_features=100)
        self.a7=torch.nn.ReLU()
        # l12
        self.drop=nn.Dropout(p=0.5)
        # l13
        self.fc2=nn.LazyLinear(out_features=6)
        self.sm=nn.Softmax()

    def forward(self,x):
        x=self.a1(self.cv1(x))
        x=self.a2(self.cv2(x))
        x=self.maxpool1(x)
        x=self.a3(self.cv3(x))
        x=self.a4(self.cv4(x))
        x=self.maxpool2(x)
        x=self.a5(self.cv5(x))
        x=self.a6(self.cv6(x))
        x=self.maxpool3(x)
        x=self.flat1(x)
        x=self.a7(self.fc1(x))
        x=self.drop(x)
        x=self.sm(self.fc2(x))
        return x

model=Net()


# compile model
criterion = nn.CrossEntropyLoss()
lr=0.0001
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
train_loader=DataLoader(dataset=dataset, batch_size=100)
validation_loader=DataLoader(dataset=vali_dataset, batch_size=100)


n_epochs=5
cost_list=[]
accuracy_list=[]
N_test=len(vali_dataset)
COST=0
import tqdm
def train_model(n_epochs):
    for epoch in tqdm.tqdm(range(n_epochs)):
        COST=0
        for x, y in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            z = model(x.float())
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST+=loss.data
        
        cost_list.append(COST)
        correct=0
        #perform a prediction on the validation  data  
        for x_test, y_test in validation_loader:
            z = model(x_test.float())
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
     
train_model(n_epochs)


print(accuracy_list)






        










        

