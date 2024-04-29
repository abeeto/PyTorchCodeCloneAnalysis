import torch                                                                                                                                                          
import torch.optim as optim                                                                                                                                           
from torch.autograd import Variable                                                                                                                                   
#training data used                                                                                                                                                                
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))                                                                                                                
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))                                                                                                              
'''
Creating our Model, which is y=2x
'''
class Model(torch.nn.Module):                                                                                                                                         
    def __init__(self):                                                                                                                                               
        super().__init__()
        #1 input and 1 output
        self.linear = torch.nn.Linear(1,1)                                                                                                                            
    def forward(self, x):                                                                                                                                             
        y_pred = self.linear(x)                                                                                                                                       
        return y_pred                                                                                                                                                 
#declaring model as a class, which is Model()                                                                                                                                                          
model = Model()                                                                                                                                                       
'''
uses Mean Square Error to find the loss between the predicted and actual data
then optimizer SGD used to find minimal loss.
'''
criterion = torch.nn.MSELoss(reduction = 'sum')                                                                                                                       
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)                                                                                                              
# Training our model, 500 times                                                                                                                                                              
for epoch in range(500):
    # passing x_data into our model to predict Y.
    y_pred = model(x_data)
    # find the loss using criterion, which is mean square error
    loss = criterion(y_pred, y_data)                                                                                                                                  
    #print(epoch, loss.item())                                                                                                                                        
    #zero the gradient and use back propagation                                                                                                                                                                
    optimizer.zero_grad()                                                                                                                                             
    loss.backward()                                                                                                                                                   
    optimizer.step()                                                                                                                                                  
#Test our model with the value 4.0                                                                                                                                                                    
test = Variable(torch.Tensor([[4.0]]))                                                                                                                                
print('Predicted Value', 4, model(test).item())                                                                                                                       
#model.eval()                                                                                                                                                         
#y_pred = model(Variable(torch.Tensor([[4.0]])))                                                                                                                      
#print(y_pred)   
