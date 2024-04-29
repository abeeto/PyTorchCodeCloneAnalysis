import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]                                                                                                                                              
y_data = [2.0, 4.0, 6.0]                                                                                                                                              
                                                                                                                                                                      
w = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value                                                                                            
                                                                                                                                                                      
# our model forward pass                                                                                                                                              
                                                                                                                                                                      
                                                                                                                                                                      
def forward(x):                                                                                                                                                       
    return x * w                                                                                                                                                      
                                                                                                                                                                      
# Loss function                                                                                                                                                       
                                                                                                                                                                      
                                                                                                                                                                      
def loss(x, y):                                                                                                                                                       
    y_pred = forward(x)                                                                                                                                               
    return (y_pred - y) * (y_pred - y)                                                                                                                                
                                                                                                                                                                      
'''                                                                                                                                                                   
# compute gradient                                                                                                                                                    
def gradient(x, y):  # d_loss/d_w                                                                                                                                     
    return 2 * x * (x * w - y)                                                                                                                                        
'''                                                                                                                                                                   
# Before training                                                                                                                                                     
print("predict (before training)",  4, forward(4).data[0])                                                                                                            
                                                                                                                                                                      
# Training loop                                                                                                                                                       
for epoch in range(20):                                                                                                                                               
    for x_val, y_val in zip(x_data, y_data):                                                                                                                          
        l = loss(x_val, y_val)                                                                                                                                        
        l.backward()                                                                                                                                                  
        print("\tgrad: ", x_val, y_val, w.grad)                                                                                                                       
        w.data = w.data - 0.01 * w.grad                                                                                                                               
                                                                                                                                                                      
        # Manually zero the gradients after updating weights                                                                                                          
        w.grad.data.zero_()                                                                                                                                           
                                                                                                                                                                      
    print("progress:", epoch, l.data)                                                                                                                                 
                                                                                                                                                                      
# After training                                                                                                                                                      
print("predict (after training)",  4, forward(4).data) 
