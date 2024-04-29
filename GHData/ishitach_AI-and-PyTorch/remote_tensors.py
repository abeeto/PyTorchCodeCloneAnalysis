import syft as sy

hook = sy.TorchHook(th)

alice = sy.VirtualWorker(hook, id="alice")
x = th.tensor([1,2,3,4,5])
x_ptr = x.send(bob,alice)

x_ptr

#(Wrapper)>[MultiPointerTensor]
#	-> (Wrapper)>[PointerTensor | me:80870295462 -> bob:62605359131]
#	-> (Wrapper)>[PointerTensor | me:94080000394 -> alice:85699948243]

x_ptr.child

#MultiPointerTensor>{'bob': (Wrapper)>[PointerTensor | me:80870295462 -> bob:62605359131], 'alice': (Wrapper)>[PointerTensor | me:94080000394 -> alice:85699948243]}

x_ptr.get(sum_results=True)
#tensor([ 2,  4,  6,  8, 10])

x = th.tensor([1,2,3,4,5]).send(bob, alice)
x.get(sum_results=True)
#tensor([ 2,  4,  6,  8, 10])


x = th.tensor([1,2,3,4,5]).send(bob)
y = th.tensor([1,1,1,1,1]).send(bob)

z = x+y
z

z = z.get()
z
#tensor([2, 3, 4, 5, 6])

x = th.tensor([1.,2,3,4,5], requires_grad = True).send(bob)
y = th.tensor([1.,1,1,1,1], requires_grad = True).send(bob)

z = (x+y).sum()
z.backward()
#(Wrapper)>[PointerTensor | me:78956402580 -> bob:5212550252]

x = x.get()
x
#tensor([1., 2., 3., 4., 5.], requires_grad=True)

x.grad
#tensor([1., 1., 1., 1., 1.])


