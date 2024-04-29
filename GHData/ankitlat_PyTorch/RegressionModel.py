#####A very simple model learning with PyTorch

###Import bare minimum torch module
import torch

### Set the seed so as to get same result with different runs
torch.manual_seed(1.0)

###data
"""Important Observation: Here we have used a model with 3 variables. Now if the number of inputs are less than 3 than it becomes
difficult for the network to learn a proper function and instead it overfits the data with some parameters (a,b,c). In order to get
proper network parameters its necessary to have at least as many data points as there are parameters in the model to be learned. 
For example in this case if we take number of data points to be less than or equal to 3 model converges but not to the expected parameters"""

#####Define number of data points training model, play by changing this
Number_Of_Datapoints = 100

#####Generate number of random datapoints for creating input dataset  
x = torch.randn((1, Number_Of_Datapoints), requires_grad=False)

####Define ground truth model parameters to be learned by training
a_gt = torch.randn(1, requires_grad=False)
b_gt = torch.randn(1, requires_grad=False)
c_gt = torch.randn(1, requires_grad=False)
#####prepare the expected ground truth output, using a quadratic model
y_gt = a_gt * x *x + b_gt*x + c_gt

#print(y_gt, x)

#########Initialize parameter values with a random value
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
c = torch.randn(1, requires_grad=True)

####Define learning parameters
epochs = 10000 ### Number of epochs
lr = 0.01      ### Learning rate

#####Creating a loop to iterate for learning the parameters
for i in range(epochs):
  #####Generate output with the initialized parameters by using all the data points in every epoch
  y = a*x*x + b*x + c
  
  ###Compute the loss which is mean square loss
  loss = torch.mean((y-y_gt).pow(2))
  
  ###Accumulate the gradient of loss with respect to the parameters
  loss.backward()
  
  ####Update the model parameters using gradient descent
  a.data = a.data - lr * a.grad.data
  b.data = b.data - lr * b.grad.data
  c.data = c.data - lr * c.grad.data
  
  """
  Initialize the grad data for each of the parameters to zero because in every backward pass (loss.backward()),
  newly calculated gradient is added to the previously calculated gradient, which we donot want and hence after updating the
  parameters these field are made zero for accumulating gradient in the next back propagation.
  """
  a.grad.data.fill_(0.0)
  b.grad.data.fill_(0.0)
  c.grad.data.fill_(0.0)
  if i % 1000 == 0:
    print('Loss {}, A {}, B {}, C{}'.format(loss.item(), a.data, b.data, c.data))

print('GT_a {}, Learned {}'.format(a_gt.data, a.data))
print('GT_b {}, Learned {}'.format(b_gt.data, b.data))
print('GT_c {}, Learned {}'.format(c_gt.data, c.data))
#print('Y_learned {}'.format(a.data*x*x + b.data * x + c.data))
