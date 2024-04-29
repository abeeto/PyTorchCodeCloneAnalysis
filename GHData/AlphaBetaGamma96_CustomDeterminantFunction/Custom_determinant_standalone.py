import torch
torch.manual_seed(0)

from functools import lru_cache
from itertools import permutations

#idx_perm, get_gamma, get_rho shamelessly taken from deepqmc! 

@lru_cache()
def idx_perm(n, r, device=torch.device('cpu')):  # noqa: B008
  idx = list(permutations(range(n), r))
  idx = torch.tensor(idx, device=device).t()
  idx = idx.view(r, *range(n, n - r, -1))
  return idx
    
def get_gamma(s):
  idx = idx_perm(s.shape[-1], 2)[-1]
  return s[..., idx].log().sum(-1).exp()

def get_rho(s):
  if(s.shape[-1] == 2):
    return s.new_ones(*s.shape,1)
  _gamma = get_gamma(s)
  idx = idx_perm(s.shape[-1], 3, s.device)[-1]
  _rho = s[..., idx].log().sum(-1).exp()
  return _rho

def _merge_on_and_off_diagonal(on_diag, off_diag):
  #store output shape, to remove and unsqueeze'd dims
  output_shape = (*off_diag.shape[:-1],off_diag.shape[-2])
  
  if(on_diag.shape[0] != off_diag.shape[0]):
    raise ValueError("Batch/Matrix dim mismatch error")
  if(len(on_diag.shape)==1 and len(off_diag.shape)==2):
    on_diag=on_diag.unsqueeze(0).unsqueeze(1)
    off_diag=off_diag.unsqueeze(0).unsqueeze(1)
  elif(len(on_diag.shape)==2 and len(off_diag.shape)==3):
    on_diag=on_diag.unsqueeze(1)
    off_diag=off_diag.unsqueeze(1)
  if(on_diag.shape[-1] != off_diag.shape[-2]):
    raise ValueError("index on_diag.shape[-1] must match off_diag.shape[-2]")
  
  dim=len(on_diag.shape)
  tmp = torch.cat((on_diag[:,:,:-1].unsqueeze(dim), \
            off_diag.view((*off_diag.shape[0:(dim-1)],off_diag.shape[-1], off_diag.shape[-2]))), dim=dim)
  res = torch.cat( (tmp.view(*off_diag.shape[0:(dim-1)], -1), on_diag[:,:,-1].unsqueeze(2)), dim=dim-1 ).view(*off_diag.shape[0:(dim-1)], on_diag.shape[-1], on_diag.shape[-1])

  return res.view(output_shape)

  
def get_off_diagonal_elements(M):
  dim=len(M.shape)-2
  if(M.shape[-2] != M.shape[-1]):
    raise ValueError("Matrix error")
    
  mask = (1 - torch.eye(M.shape[-1])) #works via broadcasting!
  return M*mask

def get_Xi_diag(M, R):
  if(M.shape != R.shape):
    raise ValueError("Shape error")
  diag_M = torch.diagonal(M, offset=0, dim1=-2, dim2=-1)
  diag_M_repeat = diag_M.unsqueeze(0).repeat(M.shape[-1], 1) #generalise to len(M.shape)-2
  MR = diag_M_repeat*R
  MR_off_diag = get_off_diagonal_elements(MR)
  return MR_off_diag.sum(dim=-1).diag_embed()
  

from torch.autograd import Function

class Determinant(Function):
  
  @staticmethod
  def forward(ctx, A):
    ctx.save_for_backward(A)
    U, S, VT = torch.linalg.svd(A)
    det = torch.det(U)*torch.det(torch.diag_embed(S))*torch.det(VT)
    return det
    
  @staticmethod
  def backward(ctx, DetBar):
    A, = ctx.saved_tensors
    return DeterminantBackward.apply(A, DetBar)
    
class DeterminantBackward(Function):

  @staticmethod
  def forward(ctx, A, DetBar):
    ctx.save_for_backward(A, DetBar)
    
    U, S, VT = torch.linalg.svd(A) 
    Abar = DetBar * torch.det(U) * torch.det(VT) * ( U @ torch.diag_embed(get_gamma(S)) @ VT )

    return Abar

  @staticmethod
  def backward(ctx, Cbar):
    A, DetBar = ctx.saved_tensors
    
    U, S, VT = torch.linalg.svd(A)
    M = VT@Cbar.transpose(-2,-1)@U
    G = get_gamma(S)
    R = _merge_on_and_off_diagonal(G, get_rho(S))

    Xi_diag = get_Xi_diag(M, R)
    Xi_off_diag = get_off_diagonal_elements(-M*R)
    Xi = Xi_diag + Xi_off_diag
    
    Agradgrad = DetBar * torch.linalg.det(U) * torch.linalg.det(VT) * U @ Xi @ VT 
    return Agradgrad, None

#==============================================================================#

def pytorch_det(matrix):
  matrices = torch.linalg.det(matrix)
  return matrices
  
def custom_det(matrix):
  matrices = Determinant.apply(matrix)
  return matrices


if(__name__=="__main__"):
  """
  #non-singular example! 
  A = 2 #number of rows/cols in matrix
  matrix = torch.randn(A,A,requires_grad=True)
  """  
  #singular example!
  matrix = torch.Tensor([[1,-2],[-3,6]])
  matrix.requires_grad=True
  
  print("matrix: ",matrix)

  custom_out = custom_det(matrix)
  custom_grad = torch.autograd.grad(custom_out, matrix, torch.ones_like(custom_out))[0]
  custom_gradgrad = torch.autograd.functional.hessian(custom_det, matrix)

  pytorch_out = pytorch_det(matrix)
  pytorch_grad = torch.autograd.grad(pytorch_out, matrix, torch.ones_like(pytorch_out))[0]
  pytorch_gradgrad = torch.autograd.functional.hessian(pytorch_det, matrix)
  
  print("\n")
  print("Output")
  print("Custom:  \n",custom_out)
  print("PyTorch: \n",pytorch_out)
  print("\n")
  print("1st-order gradient")
  print("Custom:  \n",custom_grad)
  print("PyTorch: \n",pytorch_grad)
  print("\n")
  print("2nd-order gradient")
  print("Custom:  \n",custom_gradgrad)
  print("PyTorch: \n",pytorch_gradgrad)


#Example for non-singular matrix
"""
matrix:  tensor([[ 1.5410, -0.2934],
        [-2.1788,  0.5684]], requires_grad=True)

Output
Custom:  
 tensor(0.2366, grad_fn=<DeterminantBackward>)
PyTorch: 
 tensor(0.2366, grad_fn=<SumBackward0>)


1st-order gradient
Custom:  
 tensor([[0.5684, 2.1788],
        [0.2934, 1.5410]])
PyTorch: 
 tensor([[0.5684, 2.1788],
        [0.2934, 1.5410]])


2nd-order gradient
Custom:  
 tensor([[[[ 1.4139e-08,  2.7272e-08],
          [-1.4901e-08,  1.0000e+00]],

         [[-2.7272e-08,  1.4139e-08],
          [-1.0000e+00, -1.4901e-08]]],


        [[[ 1.4901e-08, -1.0000e+00],
          [ 1.3781e-08,  5.7990e-08]],

         [[ 1.0000e+00,  1.4901e-08],
          [-5.7990e-08,  1.3781e-08]]]])
PyTorch: 
 tensor([[[[ 1.1921e-07,  0.0000e+00],
          [ 0.0000e+00,  1.0000e+00]],

         [[ 9.5367e-07,  1.9073e-06],
          [-1.0000e+00,  9.5367e-07]]],


        [[[ 1.1921e-07, -1.0000e+00],
          [ 2.9802e-08,  1.1921e-07]],

         [[ 1.0000e+00,  9.5367e-07],
          [-1.1921e-07,  0.0000e+00]]]])

"""  
  
#Example for singular matrix
"""
matrix:  tensor([[ 1., -2.],
        [-3.,  6.]], requires_grad=True)


Output
Custom:  
 tensor(1.1709e-08, grad_fn=<DeterminantBackward>)
PyTorch: 
 tensor(1.7881e-07, grad_fn=<SumBackward0>)


1st-order gradient
Custom:  
 tensor([[6.0000, 3.0000],
        [2.0000, 1.0000]])
PyTorch: 
 tensor([[6., 3.],
        [2., 1.]])


2nd-order gradient
Custom:  
 tensor([[[[ 0.,  0.],
          [ 0.,  1.]],

         [[ 0.,  0.],
          [-1.,  0.]]],


        [[[ 0., -1.],
          [ 0.,  0.]],

         [[ 1.,  0.],
          [ 0.,  0.]]]])
PyTorch: 
 tensor([[[[-1.2000e-01, -6.0000e-02],
          [-4.0000e-02,  9.8000e-01]],

         [[ 2.4000e-01,  1.2000e-01],
          [-9.2000e-01,  4.0000e-02]]],


        [[[ 1.0897e+02,  5.3486e+01],
          [ 3.6324e+01,  1.8162e+01]],

         [[ 5.4586e+01,  2.6793e+01],
          [ 1.7862e+01,  8.9310e+00]]]])

"""
  
  
  
  
  
  
  
  
