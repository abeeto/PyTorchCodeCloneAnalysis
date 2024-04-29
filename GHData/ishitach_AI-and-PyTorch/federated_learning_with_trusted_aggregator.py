pip install syft

import torch as th

import syft as sy

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
secure_worker = sy.VirtualWorker(hook, id="secure_worker")

bob.add_workers([alice,secure_worker])
alice.add_workers([bob,secure_worker])
secure_worker.add_workers([bob,alice])

data = th.tensor([[1.,1],[0,1,],[1,0],[0,0]],requires_grad=True)
target = th.tensor([[1.],[1],[0],[0]],requires_grad=True)

bob_data = data[0:2].send(bob)
bob_target = target[0:2].send(bob)

alice_data = data[2:].send(alice)
alice_target = target[2:].send(alice)

model = nn.Linear(2,1)

bobs_model = model.copy().send(bob)
alices_model = model.copy().send(alice)

bobs_opt =optim.SGD(params = bobs_model.parameters(), lr = 0.1)
alices_opt =optim.SGD(params = alices_model.parameters(), lr = 0.1)

bobs_opt.zero_grad()

for i in range(10):
  
  bobs_pred = bobs_model(bob_data)
  bobs_pred
  bob_loss = ((bobs_pred-bob_target)**2).sum()
  bob_loss.backward()
  bobs_opt.step()
  bob_loss= bob_loss.get().data
  bob_loss

  alices_pred = alices_model(alice_data)
  alices_pred
  alice_loss = ((alices_pred-alice_target)**2).sum()
  alice_loss.backward()
  alices_opt.step()
  alice_loss= alice_loss.get().data
  alice_loss
  
  
alices_model.move(secure_worker)
bobs_model.move(secure_worker)

secure_worker._objects

#{4129790193: Parameter containing:
# tensor([0.9300], requires_grad=True), 61697463307: Parameter containing:
# tensor([[-0.6651, -0.4532]], requires_grad=True), 96242995578: Parameter containing:
# tensor([-0.3225], requires_grad=True), 96389820255: Parameter containing:
# tensor([[-0.9292,  0.1925]], requires_grad=True)}

th.no_grad()
model.weight.set_(((alices_model.weight.data + bobs_model.weight.data) / 2).get())
