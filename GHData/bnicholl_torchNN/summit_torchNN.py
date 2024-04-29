#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:24:53 2018

@author: bennicholl
"""
import torch

x_test = x_test.tolist()
x_test = torch.tensor(x_test)

x = x.tolist()
y = y.tolist()

x = torch.FloatTensor(x) 
y = torch.FloatTensor(y) 


def neural_net(x,y):
    learning_rate = 0.01

    initial_dimension = x.shape[1]
    """this is our model that we will train and run test data on"""
    model = torch.nn.Sequential(
        #these jawns, nn.linear, are our weights
        torch.nn.Linear(initial_dimension, int(initial_dimension * 1.5)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension* 1.5), int(initial_dimension* 1.5)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension* 1.5), int(initial_dimension* 1.5)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension* 1.5), initial_dimension),
        torch.nn.ReLU(),
        torch.nn.Linear(initial_dimension, 1),
        torch.nn.Sigmoid()
        )
    
    loss_function = torch.nn.BCELoss()
    
    amount_of_training_examples = x.shape[0]   
    first_index = 0
    second_index = int(amount_of_training_examples / 2)
    for i in range(110000):
        if second_index <= amount_of_training_examples:
            """each vector in the param.data array represents a neuron"""
            """Forward pass: compute predicted y by passing x to the model"""
            run_x_through_graph = model(x[first_index:second_index])        
            print('forward',run_x_through_graph)
            print(run_x_through_graph.shape)
            
            """compute loss, which is the sum of -log loss divided by n training examples"""
            loss = loss_function(run_x_through_graph, y[first_index:second_index])
            #print(loss)
            
            """Zero the gradients before running the backward pass"""
            model.zero_grad()
            
            """Backward pass: compute gradient of the loss with respect to all the learnable parameters"""
            loss.backward()       
            
            first_index += 1465
            second_index += 1465
            
            """param.data represents the weights. each vector in the array is a neuron, and the matrix
            for 1 iteration is one hidden layer. param.grad.data is the corresponding gradient"""
            for param in model.parameters():
                print('param',param.data)
                print('gradient',param.grad.data)
                param.data -= learning_rate * param.grad.data
                
        else:
            first_index = 0
            second_index = int(amount_of_training_examples / 2)
        
    print(loss)
    return model

#model = neural_net(x,y)


"""this runs our testing data that we ran through our neural net"""
def run_test_data(model):
    """run test data on the newly trained model"""    
    test_data = model(x_test)    
    """turn testing data into a np array"""
    test_data = test_data.data.numpy()
    return test_data

#testing_data = run_test_data(model)


def calculate_prob_correct(x_test, y_test, testing_data):
    true_positives = 0
    true_negatives = 0
    number_correct = 0
    for index, i in enumerate(testing_data):
        if i[0] >= 0.9 and y_test[index] == 1:
            number_correct += 1
            true_positives += 1
        elif i[0] < 0.5 and y_test[index] == 0:
            number_correct += 1
            true_negatives += 1
            
    probability_of_being_correct = number_correct / len(testing_data)
    print(probability_of_being_correct)
    print(true_positives)
    print(true_negatives)
    return probability_of_being_correct        

#probability = calculate_prob_correct(x_test, y_test, testing_data)
    

def confusion_matrix(x_test, y_test, testing_data):
    unhealthy_correct = 0
    unhealthy_incorrect = 0
    healthy_correct = 0
    healthy_incorrect = 0
    for index, i in enumerate(testing_data):
        #if prediction is unhealthy and actual label is unhealthy 
        if i[0] >= 0.5 and y_test[index] == 1:
            unhealthy_correct += 1 
            
        #if prediction is unhealthy and actual label is healthy
        elif i[0] > 0.5 and y_test[index] == 0:
            unhealthy_incorrect += 1
            
        #if prediction is healthy and actual label is healthy 
        elif i[0] < 0.5 and y_test[index] == 0:
            healthy_correct += 1
            
        #if prediction is healthy and actual label is unhealthy
        elif i[0] < 0.5 and y_test[index] == 1:
            healthy_incorrect += 1
    
    #print(unhealthy_correct)
    #print(unhealthy_incorrect)
    
    #True positive
    """predicts unhealthy & label is unhealthy"""
    print(unhealthy_correct / (unhealthy_correct + unhealthy_incorrect) )
    #Flase positive
    """predicts unhealthy & label is healthy"""
    print(unhealthy_incorrect / (unhealthy_correct + unhealthy_incorrect) )
    #True positive
    """predicts healthy & label is healthy"""
    print(healthy_correct / (healthy_correct + healthy_incorrect) )
    #Flase positive
    """predicts healthy & label is unhealthy"""
    print(format(healthy_incorrect / (healthy_correct + healthy_incorrect), 'f' )) 
    
# confusion = confusion_matrix(x_test, y_test, testing_data)        
            
            
            
            
        
    

def label_data(testing_data, test_df):
    for index, i in enumerate(testing_data):
        print(index)
        if i[0] >= 0.5:
            test_df.set_value(index, 'classification', 'unhealthy')
        elif i[0] < 0.5:
            test_df.set_value(index, 'classification', 'healthy')
        
    return test_df      

#data = label_data(testing_data, test)
    


    

    


def adam_neural_net(x,y):
    learning_rate = 0.001
    initial_dimension = x.shape[1]
    """this is our model that we will train and run test data on"""
    model = torch.nn.Sequential(
        #these jawns, nn.linear, are our weights
        torch.nn.Linear(initial_dimension, int(initial_dimension * 1.5)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension* 1.5), int(initial_dimension* 1.5)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension* 1.5), int(initial_dimension* 1.5)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension * 1.5), int(initial_dimension * 1.25)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension * 1.25), int(initial_dimension * 1.25)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(initial_dimension * 1.25), initial_dimension),
        torch.nn.ReLU(),
        torch.nn.Linear(initial_dimension , 1),
        torch.nn.Sigmoid()
        )
    
    loss_function = torch.nn.BCELoss()
    adam = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    amount_of_training_examples = x.shape[0]   
    first_index = 0
    second_index = int(amount_of_training_examples / 3)
    for i in range(500000):
        if second_index <= amount_of_training_examples:
            """each vector in the param.data array represents a neuron"""
            """Forward pass: compute predicted y by passing x to the model"""
            run_x_through_graph = model(x[first_index:second_index])        
        
            """compute loss, which is the sum of -log loss divided by n training examples"""
            loss = loss_function(run_x_through_graph, y[first_index:second_index])
            #print(loss)
            
            """Zero the gradients of the adam optimizer before running the backward pass"""
            adam.zero_grad()
            
            """Backward pass: compute gradient of the loss with respect to all the learnable parameters"""
            loss.backward()       
            ###ONLY THING THAT CHANGED
            first_index += 1306
            second_index += 1306
            
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            adam.step()
            if float(loss) < 0.12:
                print(loss)
                print(i)
                return model
                
        else:
            first_index = 0
            second_index = int(amount_of_training_examples / 3)
        
    print(loss)
    return model
#model = adam_neural_net(x,y) 

    

    

    
    
    
    
    
    
    
    
    
    

    
    
        
        
        
        