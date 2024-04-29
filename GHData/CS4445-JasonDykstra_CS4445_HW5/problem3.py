import torch as th
from model import CNN
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 3: Convolutional Neural Network for Binary Image Classification (using PyTorch) (20 points)
    In this problem, we will implement a convolutional neural network (CNN) with three convolution layers with max pooling and ReLU activations, and one fully-connected layer at the end to predict the label. The classification task is that given an input gray-scale image, predict the binary label of the image (e.g., whether the image is the owner of the smartphone or not). The structure of the CNN has already been defined in the 'model.py', which is (Conv layer 1) -> ReLU -> maxpooling -> (Conv layer 2) -> ReLU -> maxpooling -> (Conv layer 3) -> ReLU -> maxpooling -> (Fully-connected layer)

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Forward Function 1) Given a convolutional neural network model (m) defined as in 'model.py' (CNN), please compute the linear logits z on a mini-batch of data samples x =[x1, x2, ... x_batch_size]. 
    ---- Inputs: --------
        * x: a mini-batch of input gray-scale images, a float torch tensor of shape (n, 1, h, w), where x[i,0] is the i-th gray-scale image in the mini-batch.
        * m: a convolutional neural network model, which is defined in "model.py" (CNN), it includes the weights and biases in all layers.
    ---- Outputs: --------
        * z: the linear logits of the last layer in CNN on a mini-batch of data samples, a float torch vector of length (n), where z[i,0] is the linear logit on the i-th data sample in the mini-batch.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z(x, m):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    z = m.forward(x)
    #########################################
    return z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_z
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_z
        --- OR ---- 
        python -m nose -v test3.py:test_compute_z
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Forward Function 2) Suppose we are given a convolutional neural network model and we have already computed the linear logits z on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average cross-entropy loss of the logistic regression model on the mini-batch of training samples. In the mean time, please also connect the global gradients of the linear logits z (dL_dz) with the loss L correctly. 
    ---- Inputs: --------
        * z: the linear logits of the last layer in CNN on a mini-batch of data samples, a float torch vector of length (n), where z[i,0] is the linear logit on the i-th data sample in the mini-batch.
        * y: the binary labels of the images in a mini-batch, a torch integer vector of length n, where y[i,0] is the label of the i-th image in the mini-batch, which can be 0 or 1.
    ---- Outputs: --------
        * L: the average binary cross entropy loss on a mini-batch of training samples, a torch float scalar.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_L(z, y):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    L = th.nn.functional.binary_cross_entropy_with_logits(z, y)
    #########################################
    return L
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_L
        --- OR ---- 
        python -m nose -v test3.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Gradient Descent) Suppose we are given a convolutional neural network model and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the weights w and biases on the mini-batch of data samples. Assume that we have already created an optimizer for all the parameters. Please update the parameters using gradient descent. After the update, the global gradients of all parameters should be set to zero. 
    ---- Inputs: --------
        * optimizer: a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (weights and bias).
    ---- Hints: --------
        * Although the parameters of the CNN model are NOT given explicitly in the input of this function, but we can assume that all the parameters are already properly configured in the optimizer. So the optimizer is configured to handle the parameters. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def update_parameters(optimizer):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    optimizer.step()
    optimizer.zero_grad()
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_update_parameters
        --- OR ---- 
        python3 -m nose -v test3.py:test_update_parameters
        --- OR ---- 
        python -m nose -v test3.py:test_update_parameters
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training CNN) Given a training dataset X (features), Y (labels) in a data loader, train a convolutional neural network model using mini-batch stochastic gradient descent: iteratively update the parametersusing the gradients on each mini-batch of data samples.  We repeat n_epoch passes over all the training samples. 
    ---- Inputs: --------
        * data_loader: a PyTorch loader of a dataset.
        * alpha: the step-size parameter of gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * m: a convolutional neural network model, which is defined in "model.py" (CNN), it includes the weights and biases in all layers.
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits and loss. 
        * Step 2 Back propagation: compute the gradients of parameters. 
        * Step 3 Gradient descent: update the parameters using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def train(data_loader, alpha=0.001, n_epoch=100):
    m = CNN() # initialize the model
    optimizer = th.optim.SGD(m.parameters(), lr=alpha) # create an SGD optimizer
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        for mini_batch in data_loader: # iterate through the dataset, with one mini-batch of random training samples (x,y) at a time
            x=mini_batch[0] # the gray-scale images in a mini-batch
            y=mini_batch[1] # the labels of the images in a mini-batch
            #########################################
            ## INSERT YOUR CODE HERE (5 points)
            z = compute_z(x, m)
            L = compute_L(z, y)
            L.backward()
            update_parameters(optimizer)
            #########################################
    return m
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_train
        --- OR ---- 
        python3 -m nose -v test3.py:test_train
        --- OR ---- 
        python -m nose -v test3.py:test_train
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the trained CNN model should have been saved into a file, named "cnn.pt", you could run a demo by typing this in the terminal: python3 demo.py
    
    '''
    


#--------------------------------------------

''' 
    TEST problem 3: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py
        --- OR ---- 
        python3 -m nose -v test3.py
        --- OR ---- 
        python -m nose -v test3.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 3 (20 points in total)--------------------- ... ok
        * (5 points) compute_z ... ok
        * (5 points) compute_L ... ok
        * (5 points) update_parameters ... ok
        * (5 points) train ... ok
        ----------------------------------------------------------------------
        Ran 4 tests in 14.2s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  batch size, the number of images in a mini-batch, an integer scalar. 
* h:  the height of each input image, an integer scalar. 
* w:  the width of each input image, an integer scalar. 
* x:  a mini-batch of input gray-scale images, a float torch tensor of shape (n, 1, h, w), where x[i,0] is the i-th gray-scale image in the mini-batch. 
* y:  the binary labels of the images in a mini-batch, a torch integer vector of length n, where y[i,0] is the label of the i-th image in the mini-batch, which can be 0 or 1. 
* m:  a convolutional neural network model, which is defined in "model.py" (CNN), it includes the weights and biases in all layers. 
* z:  the linear logits of the last layer in CNN on a mini-batch of data samples, a float torch vector of length (n), where z[i,0] is the linear logit on the i-th data sample in the mini-batch. 
* L:  the average binary cross entropy loss on a mini-batch of training samples, a torch float scalar. 
* data_loader:  a PyTorch loader of a dataset. 
* alpha:  the step-size parameter of gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (weights and bias). 

'''
#--------------------------------------------