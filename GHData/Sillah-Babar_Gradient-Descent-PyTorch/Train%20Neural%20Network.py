#!/usr/bin/env python
# coding: utf-8

# In[14]:


pip install torch


# In[5]:


import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# In[6]:


np.set_printoptions(suppress=True) 
spam_df = pd.read_csv(
    "https://hastie.su.domains/ElemStatLearn/datasets/spam.data",
    header=None,
    sep=" ")

spam_features = spam_df.iloc[:,:-1].to_numpy()
spam_labels = spam_df.iloc[:,-1].to_numpy()
# 1. feature scaling.
spam_mean = spam_features.mean(axis=0)
spam_sd = np.sqrt(spam_features.var(axis=0))
scaled_features = (spam_features-spam_mean)/spam_sd
scaled_features.mean(axis=0)
scaled_features.var(axis=0)
print(scaled_features.mean(axis=0))


# In[7]:


spam_features


# In[8]:


spam_labels


# In[9]:


class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item,:], self.labels[item]
    def __len__(self):
        return len(self.labels)


# In[10]:


#model with no hidden layers therefore simple Linear Model
class TorchLinearModel(torch.nn.Module):
    def __init__(self, num_inputs):
        super().__init__() 
        self.units_per_step=num_inputs
        self.fc1 = nn.Linear(in_features=num_inputs ,out_features=18, bias=True)
        self.fc4 =   nn.Linear(in_features=18 ,out_features=1, bias=True)
        
    def Forward(self,featureMatrix,nrow, ncol):
        x = F.relu(self.fc1(featureMatrix))
        x =self.fc4(x)
       
        return x


# In[11]:


#model with 2 hidden layers therefore a complex neural network

class TorchModel(torch.nn.Module):
    def __init__(self, num_inputs):
        super().__init__() 
        self.units_per_step=num_inputs
        self.fc1 = nn.Linear(in_features=num_inputs ,out_features=18, bias=True)
        self.fc2 =  nn.Linear(in_features=18 ,out_features=18, bias=True)
        self.fc3 =   nn.Linear(in_features=18 ,out_features=12, bias=True)
        self.fc4 =   nn.Linear(in_features=12 ,out_features=1, bias=True)
        
    def Forward(self,featureMatrix,nrow, ncol):
        x = F.relu(self.fc1(featureMatrix))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x =self.fc4(x)
       
        return x


# In[12]:


class TorchLearner():
    def __init__(self,Max_epochs,batchsize,stepsize,unitsperlayer,mod): #step size is the same as the learning rate
        self.max_epochs=Max_epochs
        self.batch_size=batchsize
        self.step_size=stepsize
        #instatntiate different model according to the value of the model where "0==complex Neural Network" "1==Simple Linear Model"
        if(mod==0):
            self.model=TorchModel(unitsperlayer)
        elif(mod==1):
            self.model=TorchLinearModel(unitsperlayer)
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.weight=torch.ones([1])
        self.loss_fun=nn.BCEWithLogitsLoss()
    def take_step(self,X, y):#where x is batch features and y is labels
        for i in range(X.shape[0]):
            predics=self.model.Forward(X[i].float(),10,10)
            tens=torch.tensor([y[i].float()])
            loss = self.loss_fun(predics, tens)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def fit(self,dl):   
        epochs=[]
        losses=[]
        count=0
        for batch_features, batch_labels in dl:
            loss=self.take_step(batch_features,batch_labels)
            #Learner.decision_function(batch_features)
            count=count+1
            if(count%6==0):
                epochs.append(count)
                losses.append(loss)
            print("Each Epoch Loss: ",loss)
        print(epochs)
        print(losses)
        plt.plot(epochs, losses)
        plt.show()
        return loss

    def decision_function(self,test_features):
        pass
        
    def predict(self,test_features):
        predics=model.Forward(test_features.float(),10,10)
        return predics
    def my_plot(self,epochs, loss):
        plt.plot(epochs, loss)


# In[13]:


class TorchLearnerCV:
    def __init__(self,sfeatures,slabels,vfeatures,vlabels):
        self.subfeatures=sfeatures
        self.sublabels=slabels
        self.valfeatures=vfeatures
        self.vallabels=vlabels
    def RunModels(self):
        print("----------------Running Code For Complex Neural Network Model-------------------------------")
        self.RunCodeForModel(0)
        print("----------------Running Code For Linear Model-------------------------------")
        self.RunCodeForModel(1)
    def RunCodeForModel(self,model_num):
        batchsize=20
        learning_rate=0.01
        for i in range(10):
            Learner=TorchLearner(max_epochs,batch_size,learning_rate,57,model_num)
        
            print("Batch Trainig for ",batchsize)
            print("Subtrain Data")
            self.fit(self.subfeatures,self.sublabels,batchsize,Learner)
            print("Validation Data")
            self.fit(self.valfeatures,self.vallabels,batchsize,Learner)
            batchsize+=100
            
    def fit(self,X,y,batch_size,Learner):
        ds = CSV(X,y)
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
        Learner.fit(dl)
    def TrainForOptimalBatch(self,batchsize):
        learning_rate=0.01
        print("----------------Running Code For Complex Neural Network Model-------------------------------")
        Learner=TorchLearner(max_epochs,batch_size,learning_rate,57,0)
        print("Batch Trainig for ",batchsize)
        print("Subtrain Data")
        self.fit(self.subfeatures,self.sublabels,batchsize,Learner)
        print("Validation Data")
        self.fit(self.valfeatures,self.vallabels,batchsize,Learner)
        print("----------------Running Code For Linear Model-------------------------------")
        Learner=TorchLearner(max_epochs,batch_size,learning_rate,57,1)
        print("Batch Trainig for ",batchsize)
        print("Subtrain Data")
        self.fit(self.subfeatures,self.sublabels,batchsize,Learner)
        print("Validation Data")
        self.fit(self.valfeatures,self.vallabels,batchsize,Learner)
      
    


# In[ ]:



np.random.seed(1)
n_folds = 5
fold_vec = np.random.randint(low=0, high=n_folds, size=spam_labels.size)
validation_fold = 0
is_set_dict = {
    "validation":fold_vec == validation_fold,
    "subtrain":fold_vec != validation_fold,
}

set_features = {}
set_labels = {}
for set_name, is_set in is_set_dict.items():
    set_features[set_name] = scaled_features[is_set,:]
    set_labels[set_name] = spam_labels[is_set]
{set_name:array.shape for set_name, array in set_features.items()}
print(set_features["subtrain"][0].shape)
print(set_labels["subtrain"][0].shape)

    
batch_size=2;
training_samples=set_features["subtrain"].shape[0]
max_epochs=training_samples/batch_size
learning_rate=0.1
#Learner=TorchLearner(max_epochs,batch_size,learning_rate,57)
#Learner.fit(set_features["subtrain"],set_labels["subtrain"])


#Learner=TorchLearner(max_epochs,batch_size,learning_rate,57,1)
#Learner.fit(set_features["validation"],set_labels["validation"])


obj=TorchLearnerCV(set_features["subtrain"],set_labels["subtrain"],set_features["validation"],set_labels["validation"])
#obj.RunModels()
#from looking at the data we figured out that batch_Size=820 is ideal
obj.TrainForOptimalBatch(120)


# In[ ]:




