"""Problem 3 - Training on MNIST"""
import numpy as np
from mytorch.nn import linear,sequential,batchnorm,activations,loss
from mytorch.optim.sgd import SGD
from mytorch import tensor
# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    model = sequential.Sequential(linear.Linear(784,20),activations.ReLU(),linear.Linear(20,10))
    optimizer = SGD(model.parameters(),lr=0.1)
    criterion = loss.CrossEntropyLoss()
    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(model,optimizer,criterion,tensor.Tensor(train_x),tensor.Tensor(train_y),tensor.Tensor(val_x),tensor.Tensor(val_y))

    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=30):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []
    
    # TODO: Implement me! (Pseudocode on writeup)
    model.eval()
    for epoch in range(num_epochs):
        indx = np.random.permutation(train_y.shape[0])
        x,y = train_x.data[indx], train_y.data[indx]
        tx,ty = np.split(x,100),np.split(y,100)
        for i, (b_x,b_y) in enumerate(zip(tx,ty)):
            optimizer.zero_grad()
            out = model.forward(tensor.Tensor(b_x))
            loss = criterion(out,tensor.Tensor(b_y))
            loss.backward()
            optimizer.step()
            if i%100==0:
                accuracy = validate(model,val_x,val_y)
                val_accuracies.append(accuracy)
                model.train()
    return val_accuracies

def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    model.eval()
    tx,ty = np.split(val_x.data,100),np.split(val_y.data,100)
    num_correct = 0
    for i,(b_x,b_y) in enumerate(zip(tx,ty)):
        out = model.forward(tensor.Tensor(b_x))
        batch_pred= np.argmax(out.data,1)
        num_correct += np.count_nonzero(batch_pred==b_y) 

    accuracy = num_correct/len(val_y.data)
    return(accuracy)

