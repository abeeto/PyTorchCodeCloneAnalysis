#Installing syft
!pip install syft

import torch as th

x = th.tensor([1,2,3,4,5])
x

import syft as sy

hook = sy.TorchHook(th)
bob = sy.VirtualWorker(hook, id="bob")

bob._objects
#{}

x = th.tensor([1,2,3,4,5])
x = x.send(bob)
bob._objects
#{27414033829: tensor([1, 2, 3, 4, 5])}

x
#(Wrapper)>[PointerTensor | me:77201033663 -> bob:27414033829]

