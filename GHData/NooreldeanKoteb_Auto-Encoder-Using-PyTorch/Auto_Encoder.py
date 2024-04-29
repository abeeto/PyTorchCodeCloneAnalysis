# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:55:52 2020

@author: Nooreldean Koteb
"""

#Auto Encoders
#Data preprocessing

#Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing movies dataset
movies = pd.read_csv('ml-1m/movies.dat',   #Dataset
                     sep = '::',           #Char to seperate values by
                     header = None,        #No headers (headers in excel files)
                     engine = 'python',
                     encoding = 'latin-1') #Help encode special charachters in dataset

#Importing users dataset
users = pd.read_csv('ml-1m/users.dat',   #Dataset
                     sep = '::',           #Char to seperate values by
                     header = None,        #No headers (headers in excel files)
                     engine = 'python',
                     encoding = 'latin-1') #Help encode special charachters in dataset

#Importing ratings dataset
ratings = pd.read_csv('ml-1m/ratings.dat',   #Dataset
                     sep = '::',           #Char to seperate values by
                     header = None,        #No headers (headers in excel files)
                     engine = 'python',
                     encoding = 'latin-1') #Help encode special charachters in dataset


#Creating the training and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


#Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

#Converting the data into an array with users in lines and movies in columns & rating as cells
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        #Gets a list of all movies & ratings, if the coresponding user id is the same
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        
        #Making no ratings to 0 for each each movie
        ratings = np.zeros(nb_movies)
        #Links movie indicies to ratings
        ratings[id_movies - 1] = id_ratings
        
        #Append movie ratings for every user
        new_data.append(list(ratings))
    
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)



#Creating the architecture of the Neural Network
#Inheriting nn.Module
class SAE(nn.Module):
    #Initializing class
    #Inherited classs variables will be placed after the ","
    def __init__(self, ):
        #Super used to inherit all classes and methods
        super(SAE, self).__init__()
        
        #Starting Encoding
        #First full connection between first input and first hidden layer
        #First input number of features
        #Seconds input number of nodes in first hidden layer, this can be tuned
        self.fc1 = nn.Linear(nb_movies, 20)
        #Second full connection
        self.fc2 = nn.Linear(20, 10)
        
        #Starting to decode
        #Third full connection
        self.fc3 = nn.Linear(10, 20)
        #Fourth full connection
        #Second input output layer
        self.fc4 = nn.Linear(20, nb_movies)
        
        #Activation function, can be changed
        self.activation = nn.Sigmoid()

    #Encoding/decoding
    #Forward propagation
    #x = input vector
    def forward(self, x):
        #New encoded vector
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        #Final decode
        x = self.fc4(x)
        
        return x


#Creating an object of our class
sae = SAE()
#Creating an object of the MSELoss class
criterion = nn.MSELoss()
#Creating an object for the Optimizer of RMSprop class, can change
#lr = learning rate & weight_Decay can be tuned
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#Training the SAE
nb_epoch = 200

for epoch in range(1, nb_epoch + 1):
    #Loss
    train_loss = 0
    #Counter (float)
    s = 0.
    
    #Loop through all users
    for id_user in range(nb_users):
        #Input vector with a new dimension (0 is the index of the new dimension)
        input_v = Variable(training_set[id_user]).unsqueeze(0)
        target = input_v.clone()
        
        #Save memory by ignoring users who didn't rate movies & optimizing code
        if torch.sum(target.data > 0) > 0:
            output = sae(input_v)
            
            #Don't compute the gradient with respect to the target
            #This optimizes the code
            target.require_grad = False
            
            #Ignore movies user didnt rate
            output[target == 0] = 0
            
            #Compute loss error
            #(predicted ratings, real ratings)
            loss = criterion(output, target)
            
            #Avg of error for movies that were rated (this considers the non-rated movies)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            
            #backward method (backpropagation) (increase or decrease)
            loss.backward()
            
            #Compute loss (RMSE)
            #State of the art error
            train_loss += np.sqrt(loss.data * mean_corrector)
            #Increment counter
            s += 1.
            
            #Apply optimizer to update the weights (backpropagation) (change by how much)
            optimizer.step()
        
    #Print training epoch and normalized train loss
    print('epoch: '+str(epoch)+'/'+str(nb_epoch)+' train loss: '+str(train_loss/s))


#Testing the SAE
#Loss
test_loss = 0
#Counter (float)
s = 0.

#Loop through all users
for id_user in range(nb_users):
    #Input vector with a new dimension (0 is the index of the new dimension)
    input_v = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    
    #Save memory by ignoring users who didn't rate movies & optimizing code
    if torch.sum(target.data > 0) > 0:
        output = sae(input_v)
        
        #Don't compute the gradient with respect to the target
        #This optimizes the code
        target.require_grad = False
        
        #Ignore movies user didnt rate
        output[target == 0] = 0
        
        #Compute loss error
        #(predicted ratings, real ratings)
        loss = criterion(output, target)
        
        #Avg of error for movies that were rated (this considers the non-rated movies)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        
        #Compute loss (RMSE)
        #State of the art error
        test_loss += np.sqrt(loss.data * mean_corrector)
        #Increment counter
        s += 1.

#Print normalized test loss
print('test loss: '+str(test_loss/s))




#same optimizations as before






