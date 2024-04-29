#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import seaborn as sns


# In[81]:


data = pd.read_excel('Input_LOG.xlsx')


# In[82]:


data = data.drop('Well', axis = 1)


# In[83]:


plt.figure(figsize=(15, 7))
sns.heatmap(data.corr())


# In[84]:


cols = list(data.columns.values)


# In[85]:


features = ['DEPTH', 'SP', 'RD', 'RXO', 'DEN', 'PE', 'CN', 'AC']
target = ['GR']


# In[86]:


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
x_sc_scaler = StandardScaler()
y_sc_scaler = StandardScaler()

X = x_sc_scaler.fit_transform(data[features])
y = y_sc_scaler.fit_transform(data[target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#data_train = pd.concat([pd.DataFrame(X_train,columns = features),pd.DataFrame(y_train, columns = target)],axis = 1)
data_train = pd.concat([pd.DataFrame(X,columns = features),pd.DataFrame(y, columns = target)],axis = 1)
#data_test = pd.concat([pd.DataFrame(X_test,columns = features),pd.DataFrame(y_test, columns = target)],axis = 1)


# In[87]:



class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,output_size,drop_out,slope):
        super(FFNN,self).__init__()
        self.slope = slope
        self.fc1 = nn.Linear(input_size,hidden_size1, bias = True)
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=slope, nonlinearity='leaky_relu')
        self.fc2 = nn.Linear(hidden_size1,hidden_size2, bias = True)
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=slope, nonlinearity='leaky_relu')
        self.fc3 = nn.Linear(hidden_size2,output_size, bias = True)
        torch.nn.init.kaiming_normal_(self.fc3.weight.data, a=slope, nonlinearity='leaky_relu')
        self.dropout = nn.Dropout(p = drop_out)
        
    def forward(self,data):
        output = F.leaky_relu(self.fc1(data),negative_slope=self.slope)
        #output = self.dropout(output)
        output = F.leaky_relu(self.fc2(output),negative_slope = self.slope)
        output = F.leaky_relu(self.fc3(output),negative_slope = self.slope)
        return output


# In[ ]:





# In[88]:



class dataset(Dataset):
    def __init__(self,data):
        super(dataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self,idx):
        feature = torch.from_numpy(self.data[:,:-1][idx]).float()
        target = torch.from_numpy(np.array(self.data[:,-1][idx])).float()
        
        sample = {"features":feature, "target": target}
        return sample


# In[89]:


input_shape = X.shape[1]
hidden_1 = 16
hidden_2 = 6
drop = 0


# In[90]:


from sklearn.model_selection import KFold


# In[91]:


folds = KFold(n_splits = 5)


# In[92]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[93]:


model = FFNN(input_shape,hidden_1,hidden_2,output_size = 1, drop_out = drop, slope = 0.01).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
epochs = 100


# In[94]:


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=False, mode = 'max')


# In[95]:


early_stopping_rounds = 15
number_epoch_without_improvements = 3
validation_loss = -np.inf


# In[96]:


for train_index, valid_index in folds.split(data_train):
    train_data, valid_data = data_train.to_numpy()[train_index],data_train.to_numpy()[valid_index]
    train_data_loader = DataLoader(dataset(train_data),batch_size=64,shuffle=True,drop_last=False)
    valid_data_loader = DataLoader(dataset(valid_data),batch_size=64,shuffle=True,drop_last=False)
    
    for epoch in range(1,epochs+1):
        model.train()
        train_loss = []
        train_preds = []
        train_truth = []
        
        for _, batch in enumerate(train_data_loader):
            X_batch = batch['features'].to(device)
            y_batch = batch['target'].to(device)
            out = model.forward(X_batch)
            loss = criterion(out, y_batch.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_truth.append(y_batch.detach().cpu().numpy())
            train_preds.append(out.view(-1).detach().cpu().numpy())
            train_loss.append(loss.item())
            
        model.eval()
        valid_loss = []
        valid_preds = []
        valid_truth = []
        
        for _,batch in enumerate(valid_data_loader):
            X_batch = batch['features'].to(device)
            y_batch = batch['target'].to(device)
            out = model.forward(X_batch)
            loss = criterion(out, y_batch.view(-1,1))
            
            valid_truth.append(y_batch.detach().cpu().numpy())
            valid_preds.append(out.view(-1).detach().cpu().numpy())
            valid_loss.append(loss.item())
        
        train_preds = np.concatenate(train_preds)
        train_truth = np.concatenate(train_truth)
        train_loss = sum(train_loss)/len(train_loss)
        r2_train = r2_score(train_truth,train_preds)
        
        valid_preds = np.concatenate(valid_preds)
        valid_truth = np.concatenate(valid_truth)
        valid_loss = sum(valid_loss)/len(valid_loss)
        r2_valid = r2_score(valid_truth,valid_preds)
        
        scheduler.step(r2_valid)
        
        if r2_valid > validation_loss:
            number_epoch_without_improvements = 0
            validation_loss = r2_valid
        else:
            number_epoch_without_improvements += 1
            
        if number_epoch_without_improvements == early_stopping_rounds:
            print(f"Early stopping at {epoch} epoch!")
            break
    
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:{3}}. Train MSELoss: {train_loss:{10}.6f}. Train R2_score: {r2_train:{6}.2f}. Validation MSELoss: {valid_loss:{10}.6f}. Validation R2_score: {r2_valid:{6}.2f}")
    print("---------------------------------------------")
            
            
        


# In[97]:


#Data from test well EI-65


# In[98]:


data_test = pd.read_excel('test_LOG.xlsx')


# In[99]:


plt.figure(figsize=(15, 7))
sns.heatmap(data_test.corr())


# In[100]:


x_t_sc_scaler = StandardScaler()
y_t_sc_scaler = StandardScaler()

X_t = x_t_sc_scaler.fit_transform(data_test[features])
y_t = y_t_sc_scaler.fit_transform(data_test[target])


# In[101]:


# Data from test well EI-65


# In[102]:


preds_test = model.forward(torch.from_numpy(X_t).float().to(device))
preds_test = preds_test.detach().cpu().numpy()


# In[103]:


print(f"R2_score on pred_test data: {r2_score(y_t, preds_test):{5}.2f}")


# In[104]:


preds_test = y_t_sc_scaler.inverse_transform(preds_test)
y_t = y_t_sc_scaler.inverse_transform(y_t)


# In[105]:


type(data_test['DEPTH'].to_numpy())


# In[109]:


plt.figure(figsize = (10,7))
plt.plot(preds_test,data_test['DEPTH'].to_numpy(),c = 'r',label = 'predictions')
plt.plot(y_t,data_test['DEPTH'].to_numpy(),c = 'b', label = 'real data')
plt.xlabel('GR')
plt.ylabel('DEPTH')
plt.gca().invert_yaxis()
plt.legend(loc='upper right')
plt.show()


# In[ ]:




