import torch as th

hook = sy.TorchHook(th)
bob = sy.VirtualWorker(hook, id="bob")

input = th.tensor([[1.,1],[0,1,],[1,0],[0,0]],requires_grad=True).send(bob)
target = th.tensor([[1.],[1],[0],[0]],requires_grad=True).send(bob)

weights = th.tensor([[0.],[0.]],requires_grad=True).send(bob)

for i in range(10):
  pred = input.mm(weights)
  loss = ((pred-target)**2).sum()
  loss.backward()
  weights.data.sub_(weights.grad*0.1)
  weights.grad *=0
  print(loss.get().data)
  
  
bob = bob.clear_objects()

bob._objects
#{}

x = th.tensor([1,2,3,4,5]).send(bob)

bob._objects
#{23369570043: tensor([1, 2, 3, 4, 5])}

del x 

bob._objects
#{}

x = "ishita"
bob._objects
#{}

for i in range(1000):
  x = th.tensor([1,2,3,4,5]).send(bob)

bob._objects

from torch import nn, optim

data = th.tensor([[1.,1],[0,1,],[1,0],[0,0]],requires_grad=True)
target = th.tensor([[1.],[1],[0],[0]],requires_grad=True)

def trains(iterations = 20):
  for iter in range(iterations):
    opt.zero_grad()
    pred= model(data)
    loss = ((pred-target)**2).sum()
    loss.backward()
    opt.step()
    print(loss.data)
trains()   

#tensor(0.2724)
#tensor(0.1221)
#tensor(0.0868)
#tensor(0.0659)
#tensor(0.0503)
#tensor(0.0385)
#tensor(0.0294)


data_bob = data[0:2].send(bob)
target_bob = target[0:2].send(bob)

data_alice = data[0:2].send(alice)
target_alice = target[0:2].send(alice)

datasets=[(data_bob, target_bob),(data_alice,target_alice)]
model = nn.Linear(2,1)
opt = optim.SGD(params = model.parameters(), lr = 0.1)

def trains(iterations = 20):
  
  model = nn.Linear(2,1)
  opt = optim.SGD(params = model.parameters(), lr = 0.1)
  
  for iter in range(iterations):
     for _data, _target in datasets:
        model = model.send(_data.location)
        opt.zero_grad()
        pred=model(_data)
        loss = ((pred-_target)**2).sum()
        loss.backward()
        opt.step()
        model =model.get()
        print(loss.get())
trains()

#tensor(0.7741, requires_grad=True)
#tensor(0.0132, requires_grad=True)
#tensor(0.0061, requires_grad=True)
#tensor(0.0051, requires_grad=True)
