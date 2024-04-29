import torch                                                                                                                                                          
from torch.autograd import Variable                                                                                                                                   
                                                                                                                                                                      
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))                                                                                                         
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))                                                                                                             
                                                                                                                                                                      
                                                                                                                                                                      
class Model(torch.nn.Module):                                                                                                                                         
                                                                                                                                                                      
    def __init__(self):                                                                                                                                               
                                                                                                                                                              
        #In the constructor we instantiate nn.Linear module                                                                                                            
                                                                                                                                                                   
        super().__init__()                                                                                                                                 
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):                                                                                                                                             
        """                                                                                                                                                           
        In the forward function we accept a Variable of input data and we must return                                                                                 
        a Variable of output data.                                                                                                                                    
        """                                                                                                                                                           
        y_pred = torch.sigmoid(self.linear(x))                                                                                                                        
        return y_pred                                                                                                                                                 
                                                                                                                                                                      
# our model                                                                                                                                                           
model = Model()                                                                                                                                                       
                                                                                                                                                                      
                                                                                                                                                                      
# Construct our loss function and an Optimizer. The call to model.parameters()                                                                                        
# in the SGD constructor will contain the learnable parameters of the two                                                                                             
# nn.Linear modules which are members of the model.                                                                                                                   
criterion = torch.nn.BCELoss(reduction='elementwise_mean')                                                                                                            
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)                                                                                                              
                                                                                                                                                                      
# Training 1000 steps                                                                                                                                                       
for epoch in range(1000):                                                                                                                                             
        # Forward pass: Compute predicted y by passing x to the model                                                                                                 
    y_pred = model(x_data)                                                                                                                                            
                                                                                                                                                                      
    # Compute and print loss                                                                                                                                          
    loss = criterion(y_pred, y_data)  
    #optional print out loss
    #print(epoch, loss.item())                                                                                                                                       
                                                                                                                                                                      
    # Zero gradients, perform a backward pass, and update the weights.                                                                                                
    optimizer.zero_grad()                                                                                                                                             
    loss.backward()                                                                                                                                                   
    optimizer.step()                                                                                                                                                  
                                                                                                                                                                      
# After training, test                                                                                                                                                      
hour_var = Variable(torch.Tensor([[1.0]]))                                                                                                                            
print("predict 1 hour ", 1.0, model(hour_var).item() > 0.5)                                                                                                           
hour_var = Variable(torch.Tensor([[7.0]]))                                                                                                                            
print("predict 7 hours", 7.0, model(hour_var).item() > 0.5)


 
  
  
#                                      A deeper Model                                     #                           
#=========================================================================================#
import torch
from torch.autograd import Variable
import numpy as np
#uses a diabetes data set with 8 different features, depending on where the file is stored.
xy = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

print(x_data.data.shape)
print(y_data.data.shape)


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(reduction='elementwise_mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
