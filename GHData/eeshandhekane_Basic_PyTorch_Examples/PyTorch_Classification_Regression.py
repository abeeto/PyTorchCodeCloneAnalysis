# Dependencies
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


# Dependencies
import torch.utils.data as Data


# Set manual seed



"""
# Saving a network
# Set manual seed
torch.manual_seed(7)


# Parameters
ITR = 100


# Define inputs and outputs
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.4*torch.rand(x.size())
x = Variable(x, requires_grad = False) # Not requiring grad
y = Variable(y, requires_grad = False) # Not requiring grad


# Define a function to save a neural network
def SaveNeuralNet(x, y):
	# Define a custom netowrk (quickly!!)	
	net = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
	)
	opt = torch.optim.SGD(net.parameters(), lr = 0.3)
	loss_fn = torch.nn.MSELoss()
	# Define training loops
	for itr in range(ITR):
		y_pred = net(x)
		loss_val = loss_fn(y_pred, y)
		opt.zero_grad()
		loss_val.backward()
		opt.step()
	# Save entire net
	torch.save(net, 'net_save.pkl')
	# Save the parameters only
	torch.save(net.state_dict(), 'net_parameters_save.pkl')


# Save an instance of the net after training
SaveNeuralNet(x, y)


# Results
# This saves the following two files in the current directory
# 	>> net_save.pkl		net_parameters_save.pkl
"""


"""
# Set manual seed
torch.manual_seed(7)


# Define inputs and outputs
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.4*torch.rand(x.size())
x = Variable(x, requires_grad = False) # Not requiring grad
y = Variable(y, requires_grad = False) # Not requiring grad


# Define a function to load the entire saved net
def LoadNeuralNet(x, y):
	net_loaded = torch.load('net_save.pkl')
	y_pred_loaded = net_loaded(x)
	print('########## 1 ##########')
	print y_pred_loaded


# Load and infer!
LoadNeuralNet(x, y)


# Results
########## 1 ##########
Variable containing:
 0.9542
 0.9376
 0.9210
 0.9044
 0.8878
 0.8712
 0.8546
 0.8380
 0.8214
 0.8048
 0.7881
 0.7715
 0.7549
 0.7383
 0.7217
 0.7051
 0.6885
 0.6719
 0.6553
 0.6387
 0.6220
 0.6054
 0.5888
 0.5722
 0.5556
 0.5390
 0.5224
 0.5058
 0.4892
 0.4726
 0.4559
 0.4393
 0.4227
 0.4061
 0.3895
 0.3729
 0.3563
 0.3397
 0.3231
 0.3065
 0.2898
 0.2732
 0.2566
 0.2400
 0.2234
 0.2068
 0.1902
 0.1736
 0.1570
 0.1404
 0.1237
 0.1071
 0.0905
 0.0887
 0.1026
 0.1232
 0.1438
 0.1644
 0.1850
 0.2056
 0.2262
 0.2468
 0.2674
 0.2880
 0.3085
 0.3291
 0.3497
 0.3703
 0.3909
 0.4115
 0.4321
 0.4527
 0.4733
 0.4939
 0.5145
 0.5351
 0.5556
 0.5762
 0.5968
 0.6174
 0.6380
 0.6586
 0.6792
 0.6998
 0.7204
 0.7410
 0.7616
 0.7821
 0.8027
 0.8233
 0.8439
 0.8645
 0.8851
 0.9057
 0.9263
 0.9469
 0.9675
 0.9881
 1.0087
 1.0292
[torch.FloatTensor of size 100x1]
"""


"""
# Set manual seed
torch.manual_seed(7)


# Define inputs and outputs
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.4*torch.rand(x.size())
x = Variable(x, requires_grad = False) # Not requiring grad
y = Variable(y, requires_grad = False) # Not requiring grad


# Define a function to load the net parameters
def LoadNeuralNetParameters(x, y):
	# First define a net prototype
	net_loaded = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
	)
	# Load the parameters
	net_loaded.load_state_dict(torch.load('net_parameters_save.pkl'))
	y_pred_loaded = net_loaded(x)
	print('########## 1 ##########')
	print y_pred_loaded


# Load and infer!
LoadNeuralNetParameters(x, y)


# Results
########## 1 ##########
Variable containing:
 0.9542
 0.9376
 0.9210
 0.9044
 0.8878
 0.8712
 0.8546
 0.8380
 0.8214
 0.8048
 0.7881
 0.7715
 0.7549
 0.7383
 0.7217
 0.7051
 0.6885
 0.6719
 0.6553
 0.6387
 0.6220
 0.6054
 0.5888
 0.5722
 0.5556
 0.5390
 0.5224
 0.5058
 0.4892
 0.4726
 0.4559
 0.4393
 0.4227
 0.4061
 0.3895
 0.3729
 0.3563
 0.3397
 0.3231
 0.3065
 0.2898
 0.2732
 0.2566
 0.2400
 0.2234
 0.2068
 0.1902
 0.1736
 0.1570
 0.1404
 0.1237
 0.1071
 0.0905
 0.0887
 0.1026
 0.1232
 0.1438
 0.1644
 0.1850
 0.2056
 0.2262
 0.2468
 0.2674
 0.2880
 0.3085
 0.3291
 0.3497
 0.3703
 0.3909
 0.4115
 0.4321
 0.4527
 0.4733
 0.4939
 0.5145
 0.5351
 0.5556
 0.5762
 0.5968
 0.6174
 0.6380
 0.6586
 0.6792
 0.6998
 0.7204
 0.7410
 0.7616
 0.7821
 0.8027
 0.8233
 0.8439
 0.8645
 0.8851
 0.9057
 0.9263
 0.9469
 0.9675
 0.9881
 1.0087
 1.0292
[torch.FloatTensor of size 100x1]
"""


"""
# Easy-to-code sequential network equivalent
# The class definition
class RegressionClass(torch.nn.Module): # Net class must build over torch.nn.Module
	# Define constructor
	def __init__(self, feat_size, hidden_size, out_size):
		super(RegressionClass, self).__init__()
		self.W_hidden = torch.nn.Linear(feat_size, hidden_size)
		self.W_out = torch.nn.Linear(hidden_size, out_size)

	# Define forward pass
	def forward(self, x):
		x1 = F.relu(self.W_hidden(x))
		y_pred = self.W_out(x1)
		return y_pred



# Create an instance of the class
net1 = RegressionClass(1, 10, 1)


# Sequential network equivalent
net2 = torch.nn.Sequential(
	torch.nn.Linear(1, 10),
	torch.nn.ReLU(),
	torch.nn.Linear(10, 1)
)


# Equivalent networks
print('1')
print(net1)
print(net2)

# Results
1
RegressionClass (
  (W_hidden): Linear (1 -> 10)
  (W_out): Linear (10 -> 1)
)
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""


"""
# Set seed 
torch.manual_seed(7) # Reproducible


# Simple data for classification
print('########## 1 ###########')
data = torch.ones(100, 2)
x_0 = torch.normal(2*data, 1) # 0-data
y_0 = torch.zeros(100) # 0-class
x_1 = torch.normal(-2*data, 1) # 1-data
y_1 = torch.ones(100) # 1-class
x = torch.cat((x_0, x_1), 0).type(torch.FloatTensor) # 32-bit float tensor
y = torch.cat((y_0, y_1), ).type(torch.LongTensor) # 64-bit int tensor
print x
print y


# Make variables trainable
print('########## 2 ###########')
x = Variable(x)
y = Variable(y)


# Define regression network: STANDARD PROCEDURE = DEFINE CLASS OF THE NET, DEFINE INITIALIZER FOR ALL LAYERS, DEFINE FORWARD PASS
class ClassificationClass(torch.nn.Module): # Net class must build over torch.nn.Module
	# Define constructor
	def __init__(self, feat_size, hidden_size, out_size):
		super(ClassificationClass, self).__init__()
		self.W_hidden = torch.nn.Linear(feat_size, hidden_size)
		self.W_out = torch.nn.Linear(hidden_size, out_size)

	# Define forward pass
	def forward(self, x):
		x1 = F.relu(self.W_hidden(x))
		y_pred = self.W_out(x1)
		return y_pred


# Define a regressor class instance
print('########## 3 ##########')
classification_net = ClassificationClass(2, 20, 2)
print(classification_net)


# Define the optimizer, loss
opt = torch.optim.SGD(classification_net.parameters(), lr = 0.3) # optimizer : Stochastic Gradient Descent
loss_fn = torch.nn.CrossEntropyLoss() # loss : Mean Squared Loss
print('########## 4 ##########')
print(opt)
print(loss_fn)


# Define training loop
print('########## 5 ##########')
ITR = 200
for itr in range(ITR):
	y_pred = classification_net(x) # Get the prediction 
	loss_val = loss_fn(y_pred, y) # Get the loss value for the iteration
	opt.zero_grad() # Reset the gradients for the training pass
	loss_val.backward() # Backpropagate the gradients
	opt.step() # Update the weights using the gradients
	if (itr + 1)%20 == 0:
		print classification_net.W_hidden.weight


# Predict the answers using the trained network
print('########## 6 ##########')
y_final_pred = classification_net(x)
y_actual = y
loss_final = loss_fn(y_final_pred, y_actual)
print loss_final
"""


"""
# Set seed
torch.manual_seed(7) # Reproducible nature


# Simple data for regression
print('########## 1 ###########')
x = torch.linspace(-1, 1, 100)
#print x # It is a tensor of size 100
x = torch.unsqueeze(x, dim = 1)
#print x # It is a tensor of size 100 x 1
y = x.pow(2) + torch.rand(x.size()) # Noisy version of y = x^2
#print y


# Convert the tensors into variables for training using gradients
print('########## 2 ##########')
x = Variable(x) # Trainable
y = Variable(y) # Trainable
#print x
#print y


# Define regression network: STANDARD PROCEDURE = DEFINE CLASS OF THE NET, DEFINE INITIALIZER FOR ALL LAYERS, DEFINE FORWARD PASS
class RegressionClass(torch.nn.Module): # Net class must build over torch.nn.Module
	# Define constructor
	def __init__(self, feat_size, hidden_size, out_size):
		super(RegressionClass, self).__init__()
		self.W_hidden = torch.nn.Linear(feat_size, hidden_size)
		self.W_out = torch.nn.Linear(hidden_size, out_size)

	# Define forward pass
	def forward(self, x):
		x1 = F.relu(self.W_hidden(x))
		y_pred = self.W_out(x1)
		return y_pred


# Define a regressor class instance
print('########## 3 ##########')
regression_net = RegressionClass(1, 20, 1)
print(regression_net)


# Define the optimizer, loss
opt = torch.optim.SGD(regression_net.parameters(), lr = 0.3) # optimizer : Stochastic Gradient Descent
loss_fn = torch.nn.MSELoss() # loss : Mean Squared Loss
print('########## 4 ##########')
print(opt)
print(loss_fn)


# Define training loop
print('########## 5 ##########')
ITR = 200
for itr in range(ITR):
	y_pred = regression_net(x) # Get the prediction 
	loss_val = loss_fn(y_pred, y) # Get the loss value for the iteration
	opt.zero_grad() # Reset the gradients for the training pass
	loss_val.backward() # Backpropagate the gradients
	opt.step() # Update the weights using the gradients
	if (itr + 1)%20 == 0:
		print regression_net.W_hidden.weight


# Predict the answers using the trained network
print('########## 6 ##########')
y_final_pred = regression_net(x)
y_actual = y
loss_final = loss_fn(y_final_pred, y_actual)
print loss_final

# Results
########## 1 ###########
########## 2 ##########
########## 3 ##########
RegressionClass (
  (W_hidden): Linear (1 -> 20)
  (W_out): Linear (20 -> 1)
)
########## 4 ##########
<torch.optim.sgd.SGD object at 0x10b10a310>
MSELoss (
)
########## 5 ##########
Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2629
-0.1744
-0.1977
-1.0529
 0.0519
-0.4360
-0.1210
 0.0653
 0.5687
 0.0414
 0.1958
 0.3621
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2535
-0.1744
-0.1977
-1.1795
 0.0519
-0.4360
-0.1210
 0.0653
 0.6997
 0.0414
 0.1958
 0.4333
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2510
-0.1744
-0.1977
-1.2655
 0.0519
-0.4360
-0.1210
 0.0653
 0.8246
 0.0414
 0.1958
 0.5033
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2499
-0.1744
-0.1977
-1.3062
 0.0519
-0.4360
-0.1210
 0.0653
 0.9067
 0.0414
 0.1958
 0.5503
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2493
-0.1744
-0.1977
-1.3272
 0.0519
-0.4360
-0.1210
 0.0653
 0.9548
 0.0414
 0.1958
 0.5775
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2490
-0.1744
-0.1977
-1.3373
 0.0519
-0.4360
-0.1210
 0.0653
 0.9812
 0.0414
 0.1958
 0.5926
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2488
-0.1744
-0.1977
-1.3410
 0.0519
-0.4360
-0.1210
 0.0653
 0.9979
 0.0414
 0.1958
 0.6022
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2486
-0.1744
-0.1977
-1.3412
 0.0519
-0.4360
-0.1210
 0.0653
 1.0098
 0.0414
 0.1958
 0.6092
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2485
-0.1744
-0.1977
-1.3406
 0.0519
-0.4360
-0.1210
 0.0653
 1.0183
 0.0414
 0.1958
 0.6150
 0.3099
[torch.FloatTensor of size 20x1]

Parameter containing:
-0.1777
 0.0079
-0.0194
 0.2313
 0.2688
 0.3139
 0.0458
-0.2483
-0.1744
-0.1977
-1.3396
 0.0519
-0.4360
-0.1210
 0.0653
 1.0258
 0.0414
 0.1958
 0.6220
 0.3099
[torch.FloatTensor of size 20x1]

########## 6 ##########
Variable containing:
1.00000e-02 *
  7.4437
[torch.FloatTensor of size 1]
"""