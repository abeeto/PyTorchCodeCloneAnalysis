import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor

from functools import lru_cache
from itertools import permutations

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

#idx_perm, get_log_gamma, get_log_rho shamelessly taken from https://github.com/deepqmc !

@lru_cache()
def idx_perm(n, r, device=torch.device('cpu')): 
  idx = list(permutations(range(n), r))
  idx = torch.tensor(idx, device=device).t() 
  idx = idx.view(r, *range(n, n - r, -1))
  return idx

def get_log_gamma(s: Tensor) -> Tensor:
  """
  Calculates the element-wise log-gamma matrix for a given S-matrix. 
  The S-matrix comes from a Singular Value Decompositon (SVD) and is defined as,

  \gamma_i = prod_{j != i} \sigma_j 
  
  So, the i-th element of the diagonal matrix is equal to the product over all S, 
  except for the i-th element of S
  
  (More info, see Appendix D page 18 of https://arxiv.org/pdf/1909.02487.pdf)
  """
  if(s.shape[-1] < 2):
    return torch.zeros_like(s)
  idx = idx_perm(s.shape[-1], 2, s.device)[-1]
  output =  s[..., idx].log().sum(-1)
  return output
  
def get_log_rho(s: Tensor) -> Tensor:
  """
  Calculates the element-wise log-rho matrix for a given S-matrix. 
  The S-matrix comes from a Singular Value Decomposition (SVD) and is defined as,
  
  \rho_ij = prod_{k != i,j} \sigma_{k}
  
  So, the i-th, j-th element of the rho matrix equals to the product over all S, 
  except for the i-th and j-the element of S. 
  
  (More info, see Appendix D page 18 of https://arxiv.org/pdf/1909.02487.pdf)
  """
  if(s.shape[-1] < 3):
    return s.new_zeros(*s.shape, 1)
  idx = idx_perm(s.shape[-1], 3, s.device)[-1]
  _rho = s[..., idx].log().sum(-1)
  return _rho
  
def _merge_on_and_off_diagonal(on_diag: Tensor, off_diag: Tensor) -> Tensor:
  """
  More details -> https://discuss.pytorch.org/t/how-can-i-merge-diagonal-and-off-diagonal-matrix-elements-into-a-single-matrix/128074/3
  
  This function takes in 2 Tensors representing the on-diagonal and off-diagonal
  elements of a batch of matrices and merges them together into a single Tensor.
  For example,
    diag = torch.tensor([11,22,33,44])
    off_diag = torch.tensor([[12,13,14],
                             [21,23,24],
                             [31,32,34],
                             [41,42,43]])
    matrix = _merge_on_and_off_diagonal(diag, off_diag)                 

    returns torch.tensor([[11,12,13,14],
                         [21,22,23,24],
                         [31,32,33,34],
                         [41,42,43,44]])

  Not the best formatting of the function but it works for the meanwhile. Apolgises here. 
  """
  output_shape = (off_diag.shape[0], off_diag.shape[1], off_diag.shape[2], off_diag.shape[-2])
  if(on_diag.shape[-1] == 1):
    return on_diag.view(output_shape)
  tmp = torch.cat( (on_diag[:,:,:-1].unsqueeze(3), \
                    off_diag.view( (off_diag.shape[0], off_diag.shape[1], off_diag.shape[-1], off_diag.shape[-2]) ) ), dim=3)
  res = torch.cat( (tmp.view(off_diag.shape[0], off_diag.shape[1], -1), on_diag[:, :, -1].unsqueeze(2)), dim=2).view(off_diag.shape[0], off_diag.shape[1], on_diag.shape[-1], on_diag.shape[-1])
  return res.view( off_diag.shape[0], off_diag.shape[1], off_diag.shape[2], off_diag.shape[-2] )

@torch.jit.script
def get_off_diagonal_elements(M: Tensor) -> Tensor:
  """
  returns a clone of the input Tensor with its diagonal elements zeroed.
  """
  res = M.clone()
  res.diagonal(dim1=-2, dim2=-1).zero_()
  return res

@torch.jit.script
def get_Xi_diag(M: Tensor, R: Tensor) -> Tensor:
  """
  A specific use case function which takes the diagonal of M and the rho matrix
  and does 

  Xi_ii = sum_{j != i} M_jj \prod_{k != ij} \sigma_k

  over a batch of matrices.
  
  (More info, see Appendix D page 18 of https://arxiv.org/pdf/1909.02487.pdf)
  """
  diag_M = torch.diagonal(M, offset=0, dim1=-2, dim2=-1)
  diag_M_repeat = diag_M.unsqueeze(2).repeat(1,1,M.shape[-1],1)
  MR = diag_M_repeat*R
  return get_off_diagonal_elements(MR).sum(dim=-1).diag_embed()
  
#==============================================================================#

"""
Here's the Custom Function, I've included a `naive' function which directly calls 
torch methods directly and follows autograd directly (without all the numerical 
stability tricks in custom version function).

The function itself is take an input shape of [B,D,N,N] where B is the batch 
dimension, D is the number of matrices per sample in the batch, and N is the 
number of rows/columns in the set of matrices.

The functions end goal is to take a determinant call of each matrix and then sum
along the D dimension such that it reduces it to a sum of determiant calls. 

However, there are numerous numerical instabilities issues that can arise. For example,
torch.linalg.det can't deal with extremely small or large determinant values 
unlike torch.linalg.slogdet which can. Unfortnately, the grad of torch.linalg.slogdet
is numerically unstable by definition. 

sgn, logabs = torch.linalg.slogdet(M)

grad_sgn_wrt_M = 0
grad_logabs_wrt_M = M.inverse().transpose(-2,-1)

So, if M is singular the gradient is ill-defined. Which can happen frequently 
in my use case.

Whereas for `torch.linalg.det` the gradient is well defined when using with a Singular
Value Decomposition,

det = torch.linalg.det(M)
grad_M = torch.linalg.det(M) * M.inverse().transpose(-2,-1).

This custom function merges the benefits of linear determinants (torch.linalg.det) and signed-log 
determiannts (torch.linalg.slogdet) into a single function call so we get the best of both worlds. 

We first take the signed-log of the inputs matrices, M, to get a set of signs and 
logabs values. We then combine those values (element-wise) to get the determinant values,

sgn, logabs = torch.linalg.slogdet(M)
detA = sgn * torch.exp(logabs)

however, this can be ill-defined if one of the logabs value is an outlier from the rest 
of them. So, we substract the maximum logabs value out of the set and add it back in later.
This shifts all the logabs values to prevent underflow
(nice explaination is here: https://www.youtube.com/watch?v=-RVM21Voo7Q). This 
then becomes a kind of signed-LogSumExp function which we then sum, and split into
its logabs and sign values (and add back in the maximum logabs value 
so our expression is equivalent).

The results ends up being a numerical stable form of,
  
  summed_dets = torch.linalg.det(M).sum(dim=-1) #M is shape [B,D,N,N] 
  global_sgn, global_logabs = torch.sign(summed_dets), (summed_dets).abs().log()

In practice my input M is defined as the element-wise product of 2 matrices, A, 
and log_envs, so M = A * torch.exp(log_envs) and the function takes in both these 
matrices as arguments and handles the backward (and double-backward) calls 
in a numerically stable manner!
"""

class GeneralisedLogSumExpEnvLogDomainStable(Function):

  @staticmethod
  def forward(ctx, matrices, log_envs):
    sgns, logabss = torch.slogdet(matrices * torch.exp(log_envs))
    max_logabs_envs = torch.max(logabss, keepdim=True, dim=-1)[0] #grab the value only (no indices)

    scaled_dets = sgns*torch.exp( logabss - max_logabs_envs )  #max subtraction

    summed_scaled_dets = scaled_dets.sum(keepdim=True, dim=-1) #sum

    global_logabs = (max_logabs_envs + (summed_scaled_dets).abs().log()).squeeze(-1) #add back in max value and take logabs

    global_sgn = summed_scaled_dets.sign().squeeze(-1) #take sign

    ctx.mark_non_differentiable(global_sgn)    #mark sgn non-differientable?
    
    ctx.save_for_backward(matrices, log_envs, global_sgn, global_logabs)
    return global_sgn, global_logabs

  @staticmethod
  def backward(ctx, grad_global_sgn, grad_global_logabs):
    matrices, log_envs, global_sgn, global_logabs = ctx.saved_tensors
    return GeneralisedLogSumExpEnvLogDomainStableBackward.apply(matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs)

class GeneralisedLogSumExpEnvLogDomainStableBackward(Function):

  @staticmethod
  def forward(ctx, matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs):

    U, S, VT = torch.linalg.svd(matrices * torch.exp(log_envs))  #all shape [B, D, A, A]
    detU, detVT = torch.linalg.det(U), torch.linalg.det(VT)      #both shape [B,D]
    log_G = get_log_gamma(S)                                     #shape [B, D, A] (just the diagonal)

    normed_G = torch.exp( log_G - global_logabs[:,None,None] )   #shape [B,D,A]
    
    U_normed_G_VT = U @ normed_G.diag_embed() @ VT 
    U_normed_G_VT_exp_log_envs = torch.exp(log_envs)*U_normed_G_VT
    sgn_prefactor = ((grad_global_logabs * global_sgn)[:,None] * detU * detVT)[:,:,None,None]
    
    dLoss_dA = sgn_prefactor * U_normed_G_VT_exp_log_envs
    dLoss_dS = matrices * dLoss_dA

    ctx.save_for_backward(U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT_exp_log_envs, grad_global_logabs,  global_sgn, global_logabs)

    return dLoss_dA, dLoss_dS

  @staticmethod
  def backward(ctx, grad_G, grad_H):
    U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
    U_normed_G_VT_exp_log_envs, grad_global_logabs, global_sgn, global_logabs = ctx.saved_tensors #get cached Tensors

    log_envs_max = torch.max(torch.max(log_envs, keepdim=True, dim=2)[0], keepdim=True, dim=3)[0]
    grad_K = (grad_G + grad_H * matrices) * torch.exp(log_envs - log_envs_max) 
    M = VT @ grad_K.transpose(-2,-1) @ U
    
    #Calculate normed rho matrices
    log_rho = get_log_rho(S)
    normed_rho_off_diag = torch.exp( log_rho - global_logabs[:,None,None,None] + log_envs_max) #scaled_rho (off-diagonal only)
    normed_rho = _merge_on_and_off_diagonal(normed_G, normed_rho_off_diag)
    
    #Calculate normed Xi matrices
    Xi_diag = get_Xi_diag(M, normed_rho)
    Xi_off_diag = get_off_diagonal_elements(-M*normed_rho)
    normed_Xi = Xi_diag + Xi_off_diag #perhaps 1 operation?

    #calculate c constant; sum(dim=(-2,-1)) is summing over kl or mn ; sum(..., dim=-1) is summing over determinants
    c = global_sgn * torch.sum( detU * detVT * torch.sum( (grad_G + matrices * grad_H ) * U_normed_G_VT_exp_log_envs, dim=(-2,-1) ), dim=-1)
    
    normed_Xi_minus_c_normed_G = (normed_Xi - c[:,None,None,None]*normed_G.diag_embed())  #don't repeat UGVT calc?
    U_Xi_c_G_VT =  U @ normed_Xi_minus_c_normed_G @ VT 
    U_Xi_c_G_VT_exp_log_envs = U_Xi_c_G_VT * torch.exp(log_envs)

    dF_dA = sgn_prefactor * (U_Xi_c_G_VT_exp_log_envs + grad_H * U_normed_G_VT_exp_log_envs)
    dF_dS = sgn_prefactor * (matrices * U_Xi_c_G_VT_exp_log_envs + (grad_G + grad_H * matrices)*U_normed_G_VT_exp_log_envs)
    
    dF_dsgn_Psi = None
    dF_dlogabs_Psi = None
    
    dF_dgrad_sgn_Psi = None 
    dF_dgrad_logabs_Psi = c
    return dF_dA, dF_dS, dF_dsgn_Psi, dF_dlogabs_Psi, dF_dgrad_sgn_Psi, dF_dgrad_logabs_Psi

"""
The `naive' method of directly writting the function without the numerical 
stability checks in the Backward and DoubleBackward calls.
"""

def NaiveLogSumExpEnvLogDomainStable(matrices, log_envs):
  sgns, logabss = torch.slogdet(matrices * torch.exp(log_envs))

  max_logabs_envs = torch.max(logabss, keepdim=True, dim=-1)[0] #grab the value only (no indices)

  scaled_dets = sgns*torch.exp( logabss - max_logabs_envs )  #max subtraction

  summed_scaled_dets = scaled_dets.sum(keepdim=True, dim=-1) #sum

  global_logabs = (max_logabs_envs + (summed_scaled_dets).abs().log()).squeeze(-1) #add back in max value and take logabs

  global_sgn = summed_scaled_dets.sign().squeeze(-1) #take sign

  return global_sgn, global_logabs

#==============================================================================#

def naive_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = NaiveLogSumExpEnvLogDomainStable(A, log_envs)
  return global_sgn, global_logabs

def custom_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = GeneralisedLogSumExpEnvLogDomainStable.apply(A, log_envs)
  return global_sgn, global_logabs

nsamples = 1#00
num_dets = 4
num_inputs = 6

A = torch.randn(nsamples, num_dets, num_inputs, num_inputs, requires_grad=True)
log_envs = torch.rand(nsamples, num_dets, num_inputs, num_inputs, requires_grad=True)

custom_out = custom_summed_det(A, log_envs)
naive_out = naive_summed_det(A, log_envs)

sign_forward_check = torch.allclose(naive_out[0], custom_out[0])
logabs_forward_check = torch.allclose(naive_out[1], custom_out[1])

backward_check = torch.autograd.gradcheck(func=custom_summed_det, inputs=(A, log_envs), raise_exception=False)

double_backward_check = torch.autograd.gradgradcheck(func=custom_summed_det, inputs=(A, log_envs), raise_exception=False)


print("forward check (sign): ", sign_forward_check)
print("forward check (logabs): ",logabs_forward_check)
print("gradcheck: ",backward_check)
print("gradgradcheck: ",double_backward_check)

def naive_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = NaiveLogSumExpEnvLogDomainStable(A, log_envs)
  return global_logabs

def custom_summed_det(A: Tensor, log_envs: Tensor) -> Tensor:
  global_sgn, global_logabs = GeneralisedLogSumExpEnvLogDomainStable.apply(A, log_envs)
  return global_logabs

naive_jac = torch.autograd.functional.jacobian(func=naive_summed_det, inputs=(A, log_envs))
naive_hess = torch.autograd.functional.hessian(func=naive_summed_det, inputs=(A, log_envs))

custom_jac = torch.autograd.functional.jacobian(func=custom_summed_det, inputs=(A, log_envs))
custom_hess = torch.autograd.functional.hessian(func=custom_summed_det, inputs=(A, log_envs))

print("\nCheck custom function vs naive function")
print("For larger matrices, in D or N, the difference will become more apparent\n")

for i in range(2):
  print("Jacobian check: ", i, torch.allclose(naive_jac[i], custom_jac[i]))

for i in range(2):
  for j in range(2):
    print("Hessian check: ",i,j,torch.allclose(naive_hess[i][j], custom_hess[i][j]))    


"""
Output: 

forward check (sign):  True
forward check (logabs):  True
gradcheck:  False
gradgradcheck:  True

Check custom function vs naive function
For larger matrices, in D or N, the difference will become more apparent

Jacobian check:  0 True
Jacobian check:  1 True
Hessian check:  0 0 True
Hessian check:  0 1 True
Hessian check:  1 0 True
Hessian check:  1 1 True

For the 1st backward check it will thrown an error for raise_exception=True

backward_check = torch.autograd.gradcheck(func=custom_summed_det, inputs=(A, log_envs), raise_exception=True)

results in,

torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,

output 0 is the global sign value and the 0-th input is the A input matrix, the gradient 
of a sign operation is zero by defintion. So, it's interesting to see the gradcheck fail here...
(Unless I'm mistaken of course).
"""
