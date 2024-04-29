import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
	def __init__(self,D_in,H,D_out):
		'''
		In the constructor we instiantiate two nn.Linear modules and assign
		them as member variables
		'''
		super(TwoLayerNet,self).__init__()
		self.linear1=torch.nn.Linear(D_in,H)
		self.linear2=torch.nn.Linear(H,D_out)

	def forward(self,x):
		'''
		In the forward function we accept a Variabel of input data and
		we must return a Variabel of output data.We can also use Modules
		defined in the constructors as well as arbitrary operators on Variables.
		'''
		h_relu=self.linear1(x).clamp(min=0)
		y_pred=self.linear2(h_relu)
		return y_pred

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)

model=TwoLayerNet(D_in,H,D_out)

criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
 
for t in range(500):
	y_pred=model(x)
	loss=criterion(y_pred,y)
	print(t,loss.data[0])

	#zero gradients,perform a backward pass, adn update the weights
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()