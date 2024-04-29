import torch

#=========================================================#
#                                     Initializing Tensor                                                #
#=========================================================#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tensor = torch.tensor([[1, 2, 3], [5, 6, 7]], dtype=torch.float32,
                         device=device, requires_grad=True)

# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)


#=========================================================#
#                        Other common initialization methods                               #
#=========================================================#
x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5)
x = torch.arange(start=0, end=10, step=2)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
x = torch.empty(size=(1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3))
# print(x)

#=========================================================#
#    How to initialize & convert tensors to other types(init, float, double)      #
#=========================================================#
tensor = torch.arange(4)
print(tensor.bool())     # boolean True/False
print(tensor.short())   # init16
print(tensor.long())   # init64 (Important)
print(tensor.half())   # float16
print(tensor.float())   # float32 (Important)
print(tensor.double())   # float64




