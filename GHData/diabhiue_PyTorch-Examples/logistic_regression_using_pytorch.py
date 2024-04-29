import torch
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt

#Input data and output data
x_data = Variable(torch.Tensor([[-2.0], [2.0], [5.0], [-5.0], [-0.01],[0.00000000001], [-0.00000000001], [0.01], [7.0], [-7.0], [10000.0], [-10000.0]]))
y_data = Variable(torch.Tensor([[0.0], [1.0], [1.0], [0.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]]))

#Model
class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.linear = torch.nn.Linear(1, 1)	#linear model

	def forward(self, x):
		return F.sigmoid(self.linear(x))	#y_pred

#Instance creation
model = Model()

#Binary Cross Entropy for loss
criterion = torch.nn.BCELoss(size_average=True)

#Stochastic Gradient Descent for opimization of model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Lists created for graph plotting
epoch_list = list()
loss_list = list()

#Iterating through each epoch
for epoch in range(1000):
	y_pred = model(x_data)				#input data fit into model to give predicted output data

	loss = criterion(y_pred, y_data)	#loss calculation using BCE
	print(epoch, loss.data[0])			

	epoch_list.append(epoch)
	loss_list.append(loss.data[0])

	optimizer.zero_grad()				
	loss.backward()						#creates computation graph for loss with respect to all dependent variables
	optimizer.step()					#updates weight 'w' using SGD

plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#Test input data
x_test = Variable(torch.Tensor([[1.5], [-1.5], [-0.0001], [0.0001], [5000]]))

#Predicts output data and converts it in binary depending upon data
y_test_pred = model.forward(x_test).data > 0.5
print("Predicted output for test data is {}".format(y_test_pred))