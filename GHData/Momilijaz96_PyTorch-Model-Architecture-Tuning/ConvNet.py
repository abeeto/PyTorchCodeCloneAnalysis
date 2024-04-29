import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        self.out1=nn.Linear(100,10)
        self.out2=nn.Linear(1000,10)

        self.fcl1=nn.Linear(1*28*28,100)
        self.fcl2=nn.Linear(40*18*18,100)
        self.fcl2=nn.Linear(40*18*18,100)
        self.fcl3=nn.Linear(100,100)
        self.fcl4=nn.Linear(40*18*18,1000)
        self.fcl5=nn.Linear(1000,1000)
            
        self.conv1=nn.Conv2d(1,40,5)
        self.conv2=nn.Conv2d(40,40,5)
            
        self.drop_layer=nn.Dropout2d(p=0.5)
                # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
            print("Selected Mode 1")

        elif mode == 2:
            self.forward = self.model_2
            print("Selected Mode 2")

        elif mode == 3:
            self.forward = self.model_3
            print("Selected Mode 3")

        elif mode == 4:
            
            self.forward = self.model_4
            print("Selected Mode 4")

        elif mode == 5:
            
            self.forward = self.model_5
            print("Selected Mode 5")

        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        X=X.view(-1,np.prod(X.size()[1:]))# [...., final_dim=input nodes of fcl]
        X=torch.sigmoid(self.fcl1(X))
        return F.log_softmax(self.out1(X),dim=1)

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        X=F.max_pool2d(F.relu(self.conv1(X)),2,1)
        X=F.max_pool2d(F.relu(self.conv2(X)),2,1)
        X=X.view(-1,np.prod(X.size()[1:]))# [...., final_dim=input nodes of fcl]
        X=torch.sigmoid(self.fcl2(X))
        return F.log_softmax(self.out1(X),dim=1)

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        X=F.max_pool2d(F.relu(self.conv1(X)),2,1)
        X=F.max_pool2d(F.relu(self.conv2(X)),2,1)
        X=X.view(-1,np.prod(X.size()[1:]))# [...., final_dim=input nodes of fcl]
        X= F.relu(self.fcl2(X))
        return F.log_softmax(self.out1(X),dim=1)

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        X=F.max_pool2d(F.relu(self.conv1(X)),2,1)
        X=F.max_pool2d(F.relu(self.conv2(X)),2,1)
        X=X.view(-1,np.prod(X.size()[1:]))# [...., final_dim=input nodes of fcl]
        X=F.relu(self.fcl2(X))
        X=F.relu(self.fcl3(X))
        return F.log_softmax(self.out1(X),dim=1)

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        X=F.relu(F.max_pool2d((self.conv1(X)),2,1))
        X=F.relu(F.max_pool2d((self.conv2(X)),2,1))
        X=X.view(-1,np.prod(X.size()[1:]))# [...., final_dim=input nodes of fcl]
        X=F.relu(self.fcl4(X))
        X=F.relu(self.fcl5(X))
        X=self.drop_layer(X)
        return F.log_softmax(self.out2(X),dim=1)
    
    
