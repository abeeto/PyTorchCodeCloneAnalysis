import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import data_processing as dp
import numpy as np
import pandas as pd
from math import log


#Skill_numpy = np.loadtxt('UCI_REPO/Data_Cortex_Nuclear.xls', delimiter=',', dtype=np.float32, skiprows=1)
input_data = (dp.readUCRdata('dermatology'))
input_data = input_data.astype(np.float32)
Skill_tensor = torch.from_numpy(input_data)
Skill_tensor = torch.nn.functional.normalize(Skill_tensor, p=2.0, dim=1, eps=1e-12, out=None)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(Skill_tensor.shape[1],15),
            #nn.Dropout(p=0.5),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(15, 10),
            #nn.Dropout(p=0.5),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            #nn.Dropout(p=0.5),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(10, 15),
            #nn.Dropout(p=0.5),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(15,Skill_tensor.shape[1]),
            nn.ReLU()
            #nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def model_training(num_of_epoch):

    data_loader = torch.utils.data.DataLoader(dataset=Skill_tensor,
                                              batch_size=64,
                                              shuffle=True)
    ## manually fixing the seed
    torch.manual_seed(7)
    
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    num_epochs = num_of_epoch
    Error = []
  
    

    for epoch in range(num_epochs):
        for batch in data_loader:
            recon = model(batch)
            loss = criterion(recon, batch) 
            
            ## Regularization
            # Compute L1 loss component
            l1_weight = 1.0
            #l1_parameters = []
            
            # To access the model weights
            state_dict = model.state_dict()
            l1_reg = torch.tensor(0.)
            
            for name,parameters in state_dict.items():
                if not "weight" in name or "decoder" in name:
                    #print(name)
                    continue
                    
                l1_reg += torch.norm(parameters,1)
                
                #W =  parameters.numpy()
                #wm = torch.from_numpy(W)


            
            #l1_norm = sum(p.abs().sum() for p in wm)
 
            loss = loss + l1_weight * l1_reg
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Collecting log loss at each epoch
        Error.append(-np.log(loss.item()))

        # access names and params 
        for name, param in state_dict.items():

                    # ignore all bias terms and the decoder weights
                    # because encoder weight = decoder weights
                    if not "weight" in name or "decoder" in name:
                        continue

                    #Only encoder weights
                    W =  param.numpy()
                    # Apply any transformation on W here
                    # transform it back to torch tensor
                    wm = torch.from_numpy(W)

                    # Feed the transformed weight back to model states
                    state_dict[name].copy_(wm)
                    
    return Error,state_dict