import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):

    def forward(self,input):
	    self.save_for_backward(input)
	    return input.clamp(min=0)

    def backward(self,grad_output):
        input,=self.saved_tensors
        grad_input=grad_output.clone()
        grad_input[input<0]=0
        return grad_input


dtype=torch.FloatTensor

N,D_in,H,D_out=64,1000,100,10

x=Variable(torch.randn(N,D_in).type(dtype),requires_grad=False)
y=Variable(torch.randn(N,D_out).type(dtype),requires_grad=False)

#create random Tensors for weights and wrap them in Variables
w1=Variable(torch.randn(D_in,H).type(dtype),requires_grad=True)
w2=Variable(torch.randn(H,D_out).type(dtype),requires_grad=True)

learning_rate=1e-6
for t in range(500):
	#construct an instance of our MyReLu class to use in our network
	relu=MyReLU()

	#Forward pass:compute predicted y using operations on Variables
	y_pred=relu(x.mm(w1)).mm(w2)

	#compute and print loss
	loss=(y_pred-y).pow(2).sum()
	print(t,loss.data[0])

	loss.backward()

	#update weights using gradient descent
	w1.data-=learning_rate*w1.grad.data
	w2.data-=learning_rate*w2.grad.data

	#manully zero the gradients after updating weights
	w1.grad.data.zero_()
	w2.grad.data.zero_()
