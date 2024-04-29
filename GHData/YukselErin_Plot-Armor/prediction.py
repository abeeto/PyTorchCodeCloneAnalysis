import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import runpy

#This will be the model we are training. One layer linear model with sigmoid function at the output. We needed to add the sigmoid function to work with binary class problem.
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class Predictor(object):
    #This is the function we will use to train a model with given dataset. It takes the size of the training set and the training data as two tensors.
    #The model is logistic regression with binary cross enropy as loss function.
    #The function returns both the model and the last batch's loss value, in order to prevent badly shuffled data to result in a high loss model. 
    def train(self,size,inputs,targets):
        model = LogisticRegression(3, 2)                                #We declare a logistic regression model with 3 inputs and 2 outputs
        trainingDS = TensorDataset(inputs, targets)                     #We take our input and target tensors and create a dataset
        batch_size = 5                                                  #This is the amount of training data to go through at each iteration
        trainingDL = DataLoader(trainingDS, batch_size, shuffle=True)   #We create the loader for the data and enable shuffle for randomized distribution
        loss_func = torch.nn.BCELoss()                                  #As our loss function we will use binary cross enropy function
        opt = torch.optim.SGD(model.parameters(), lr=0.01)              #This is the gradient descent function from torch library
        for epoch in range(0,size):
            for xb, yb in trainingDL:
                prediction = model(xb)                                  #We use our model to get a prediction
                loss = loss_func(prediction, yb)                        #We calculate how bad the prediction is via loss function
                loss.backward()                                         #This computes the gradients 
                opt.step()                                              #And this uses the gradients to update the weights and biases
                opt.zero_grad()                                         #We need to zero the current gradient information in order to calculate new ones in nex iteration
            if (epoch +1) % 10 ==0: 
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, size, loss.item()))
        return model,loss.item()                                                    #After we trained for the specifed time we return the trained model, I am also returning the last predicted loss function

    def getdata(self,filename):                                         
        with open(str(filename),'rb') as file:
            inputs = np.load(file)
            targets = np.load(file)
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)
        inputs = np.delete(inputs, 0, 0)
        targets = np.delete(targets, 0, 0)
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        return inputs, targets

predictor = Predictor()
inputs1, targets1 = predictor.getdata("batch1.npy")
inputs2, targets2 = predictor.getdata("batch2.npy")

loss = 10
while loss > 3:
    model,loss = predictor.train(100,inputs1,targets1)
    model,loss = predictor.train(100,inputs2,targets2)




while True:
    print("Predict the outcome of game for: ")
    agro = np.float32(input("Enemy aggresiveness: "))
    health = np.float32(input("Player health: "))
    power = np.float32(input("Player power: "))
    testing = np.array([agro,health,power])
    testing = torch.from_numpy(testing)
    if torch.tensor(1) == torch.argmax(model(testing),dim=0):
        print("Player victory is predicted!")
    else:
        print("Enemy victory is predicted!")
    params = str(int(agro)) +","+ str(int(health))+ ","+ str(int(power))
    runpy.run_path(path_name='game1.py ', run_name= params)
    
    
    


