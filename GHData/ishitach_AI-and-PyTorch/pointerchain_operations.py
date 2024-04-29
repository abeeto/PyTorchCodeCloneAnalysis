x = th.tensor([1,2,3,4,5]).send(bob)
x = x.send(alice)

bob._objects

alice._objects

y = x+x
y

bob._objects

alice._objects

jon = sy.VirtualWorker(hook, id="jon")
x = th.tensor([1,2,3,4,5]).send(bob).send(alice)
y = th.tensor([1,2,3,4,5]).send(bob).send(jon)
x = x.get()

alice.clear_objects()
bob.clear_objects()

x = th.tensor([1,2,3,4,5]).send(bob).send(alice)
x.remote_get()

x.move(bob)
