import torch
import torch.autograd as autograd
torch.manual_seed(1)

# ================= Torch Tensors =================
'''
Torch tensors represent finite-dimensional arrays. 
We can say they are generalized matrices. They are
the most important PyTorch (and Deep Learning in general)
data structure. They allow us to store, and perform varied 
computations on data (matrix multiplication, element-wise
multiplication, addition, concatenation, etc).
'''
print("Torch Tensors")
V = [1., 2., 3.]
V_ten = torch.tensor(V)
print("V_ten:", V_ten)

M = [ [1., 2., 3.], [4., 5., 6] ]
M_ten = torch.tensor(M)
print("M_ten:", M_ten)

T = [ [[1., 2., 3.], [4., 5., 6.]],
      [ [7., 8., 9.], [10., 11., 12.] ]]
T_ten = torch.tensor(T)
print("T_ten:", T_ten)
print("\n")

# ========= Accessing values of tensor entries ===============
'''
To obtain a value (which can be sub-tensor) with indices x, y, z... 
of a tensor Ten, call Ten[x][y][z]...
For tensors with one value can use Ten.item() to extract the 
single value from the tensor
'''
print("Accessing values of tensor entries")
print("V_ten[0]:",V_ten[0])
print("V_ten.item():", V_ten[0].item() )
print("M_ten[0]:", M_ten[0])
print("T_ten[0]:", T_ten[0])
print("\n")
# ================== Tensor concatenation =======================
'''
To concatenate two tensors, ie. to stick them in one of the dimensions
(you can think of it as taking two n-dimensional rectangles and sticking 
them to each other by one side),
use torch.cat(tensor1, tensor2, axis= theDimension). By default, torch
concatenates along the first dimension (row).

BTW: torch.randn(n,m) samples a tensor of dimensions n and m from 
the normal distribution.
'''
print("Tensor concatenation")
x1 = torch.randn(2, 5)
x2 = torch.randn(3, 5)
z1 = torch.cat([x1, x2])
print("Concatenation along the default dimension")
print("z1: ", z1)

y1 = torch.randn(2, 3)
y2 = torch.randn(2, 5)
z2 = torch.cat( [y1, y2], 1)
print("Concatenation along the second(number 1) dimension")
print("z2:", z2)
print("\n")

# =================== Tensor reshaping ===========================
'''
Torch enables us to reshape tensors. This is particularly useful, as
we often need to adjust our data to feed different models. We do
it by 'view' method.

If the among the dimensions you want to pass to view method, one 
obviously follows from the others (you want to change a 3x4 tensor
into 12x1, then if you specify the first dimension in view, ie 12,
from 3x4/12=1, it follows then the second dimension is 1. Passing -1
to one of dimension arguments is equivalent to saying to Torch:
'you can figure this out'.
'''
print("Tensor reshaping")
x = torch.randn(2, 3, 4)
print(x)

print(x.view(2, 12))
print(x.view(2, -1) )
print('\n')

# ================ Automatic Differentiation =====================
'''
When creating a tensor, you can specify requires_grad=True, which
gives a signal to Torch that the new tensor is a Torch variable.
Since then, Torch will keep track of it, automatically computing 
gradients of new variables with respect to the tensor.

You can check how each such variable is computed accessing its 
grad_fn attribute.

Once you have a single value tensor, you can perform backpropagation. 
It will compute gradients with respect to all variables along the 
way to it.
'''
print("Automatic Differentiation")
x= torch.tensor([1., 2., 3.], requires_grad= True)
y= torch.tensor([4., 5., 6.], requires_grad= True)
z= x+y
print("z.grad_fn: ", z.grad_fn)

s= z.sum()
print("s: ", s)
print("s.grad_fn:", s.grad_fn)

s.backward()
print("Gradient with respect to x:", x.grad)
print("\n")

# ================ Switching varibles on/off and torch.no_grad() ==================
'''
For a tensor Tensor, calling Tensor.requires_grad_() changes its 
value of requires_grad argument. This means that something may start, or 
stop being a tensor variable.

To copy a tensor's values without gradient, you can use
detach() method.

Another essential methos is torch.no_grad(). It enables you to perform
computations using your variables without computing their gradients. 
It is useful, for example, when you have already trained your model, and
now you want to use it.
'''
print("Switch variables on/off and torch.no_grad()")
x = torch.randn(2,2)
y = torch.randn(2,2)
print("x and y and their requires_grad:", x.requires_grad, y.requires_grad)
z = x+y
print("z.grad_fn:", z.grad_fn)
print("\n")

x.requires_grad_()
y.requires_grad_()
print("x and y and their requires_grad:", x.requires_grad, y.requires_grad)
z = x+y
print("z.grad_fn:", z.grad_fn)

new_z = z.detach()
print("new_z:", new_z)
print("new_z.grad_fn:", new_z)

print("x.requires_grad:", x.requires_grad)
print("x**2.requires_grad:", (x**2).requires_grad)

with torch.no_grad():
    print("x.requires_grad:", x.requires_grad)
    print("x**2.requires_grad:", (x ** 2).requires_grad)