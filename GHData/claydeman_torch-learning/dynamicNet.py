import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
	def __init__(self,D_in,H,D_out):
		super(DynamicNet,self).__init__()
		self.input_linear=torch.nn.Linear(D_in,H)
		self.middle_linear=torch.nn.Linear(H,H)
		self.output_linear=torch.nn.Linear(H,D_out)
	def forward(self,x):
		'''
		for the forward pass of the model.We randomly choose either 0,1,2,or 3 and
		reuse the middle_linear Module that many times to compute hidden layer
		representations
		'''
		h_relu=self.input_linear(x).clamp(min=0)
		for _ in range(random.randint(0,3)):
			h_relu=self.middle_linear(h_relu).clamp(min=0)			
		y_pred=self.output_linear(h_relu)
		return y_pred

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)

model=DynamicNet(D_in,H,D_out)

criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)

for t in range(500):
	y_pred=model(x)

	loss=criterion(y_pred,y)
	print(t,loss.data[0])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()