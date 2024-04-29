import torch
from torch.autograd import Variable

N,D_in,H,D_out=64,1000,100,10

#create random Tensors to hold inputs and outputs, and wrap them in Variables
x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)

#use the nn package to define our model as a sequence of layers.
model=torch.nn.Sequential(
	torch.nn.Linear(D_in,H),
	torch.nn.ReLU(),
	torch.nn.Linear(H,D_out),
)
#The nn package also contains difinitions of popular loss functions
loss_fn=torch.nn.MSELoss(size_average=False)

learning_rate=1e-4

for t in range(500):
	y_pred=model(x)
	#compute and print loss.
	loss=loss_fn(y_pred,y)
	print(t,loss.data[0])

	#zero the gradients before running the backward pass
	model.zero_grad()

	loss.backward()
	for param in model.parameters():
		param.data-=learning_rate*param.grad.data