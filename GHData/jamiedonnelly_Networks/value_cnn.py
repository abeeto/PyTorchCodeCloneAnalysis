import numpy as np 
import logging 
import torch 
from torch import nn
from torch.functional import F
from tqdm import tqdm 


def value_loss(target, output):
    """
      Function to calculate empirical risk from misprediction, operation on spatial grids. Squared loss is scaled by the value for each
      cell to calculate a value-adjusted loss. 
    """
    global values 
    squared_diff = torch.square(target - output)
    loss = values * squared_diff
    return torch.mean(loss)

class ValueNetwork(torch.nn.Module):

    def __init__(self,input_dim,output_dim):
        super(ValueNetwork,self).__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(1,32,1)
        self.conv2 = nn.Conv1d(32,128,1)
        self.fc1 = nn.Linear(128*self.input_dim,32)
        self.fc2 = nn.Linear(32,256)
        self.fc3 = nn.Linear(256,512)
        self.fc4 = nn.Linear(512,self.output_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.fc4(x)
        return output
        
        

if __name__=="__main__":
    
    logging.basicConfig(filename='network_train.log', level=logging.INFO, 
    format="%(asctime)s: %(levelname)s: %(message)s")

    root = '...'

    with open(f"{root}/data/X.npy",'rb') as f:
        X = torch.Tensor(np.load(f))

    with open(f"{root}/data/y.npy",'rb') as f:
        y = torch.Tensor(np.load(f))

    with open(".../ValueMatrix.npy",'rb') as f:
        values = np.load(f).flatten()
        values = torch.Tensor(values)

     network = ValueNetwork(x_train.shape[2], ytr.shape[1])

        epochs = 5
        batch_size = 64
        loss = value_loss
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        for epoch in range(epochs):

            logging.info(f"Beginning training epoch...{epoch}...")

            running_loss = 0.0

            permutation = torch.randperm(x_train.size()[0])

            for batch in tqdm(range(0, x_train.size()[0], batch_size)):
                
                optimizer.zero_grad()

                indices = permutation[batch:batch+batch_size]

                batch_x, batch_y = x_train[indices], ytr[indices]

                outputs = network.forward(batch_x)

                mse = loss(batch_y, outputs)

                running_loss += mse

                mse.backward()

                optimizer.step()

            logging.info(f"Running loss...{running_loss/len(range(0, x_train.size()[0], batch_size)):.4f}")



