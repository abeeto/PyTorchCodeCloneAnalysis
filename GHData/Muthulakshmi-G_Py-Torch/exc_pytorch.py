import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

#2matrices 2X3
d=[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
d=torch.Tensor(d)
print("shape of the tensor:",d.size())
print(d)
#adding two matrices
add=d[0]+d[1]

print("adding up two matrices",add)
#reshaping the d
print(d.view(2,-1))

#autograd-we can compute gradients automatically
#d is a tensor not node.to create a node.
x=autograd.Variable(d,requires_grad=True)
print("the nodes data is the tensor:",x.data.size())

print("the node is gradient empty at creation:",x.grad)

#do operation on the node to make computationala graph
y=x+1
z=x+y
s=z.sum()


s.backward()
print("the variable now has gradients:",x.grad)

#linear transformation of a matrix 2X5 matrix 2x3
linear_map=nn.Linear(5,3)
print("using randomly intilaized params:",linear_map.parameters)

#2 examples with 5 features
data=torch.randn(2,5)#training
y=autograd.Variable(torch.randn(2,3))#target
#make a node
x=autograd.Variable(data,requires_grad=True)

#apply transformation to a node creates computational graph
a=linear_map(x)
z=F.relu(a)

o=F.softmax(z)

print("output of softmax as probability distribution:",o.data.view(1,-1))

loss_func=nn.MSELoss()
L=loss_func(z,y)#loss between output and target

print("loss:",L)

#forward()=>input and output as Variable()

#we pass input through the layer,perform operation on input by using parameters and returns the output.input need to be autograd.Variable()

class Log_reg_classifier(nn.Module):
    def __init__(self,in_size,out_size):
        super(Log_reg_classifier,self).__init__()
        self.linear=nn.Linear(in_size,out_size)#layer parameters
    def forward(self,vect):
        return F.log_softmax(self.linear(vect))

#optimization
optimizer=torch.optim.SGD(linear_map.parameters(),lr=0.01)

optimizer.zero_grad() #make gradients zero
L.backward()
optimizer.step()
print(L)

print("==================")
model=Log_reg_classifier(10,2)

loss_func=nn.MSELoss()

optimizer=optim.SGD(model.parameters(),lr=0.001)

#send data through tha model in minibatches for 10 epochs

for epoch in range(10):
    for minibatch,target in data:
        model.zero_grad()#zero for each batches
        #forward pass
        out=model(autograd.Variable(minibatch))
        #backward pass
        L=loss_func(out,target)#calculate loss
        L.backward()#calculate gradients

        optimizer.step()#make an update step
