from __future__ import print_function
import torch
import numpy

#torch.manual_seed(seed)
#torch.initial_seed()
#torch.get_rng_state()
#torch.set_rng_state(new_state)
# torch.default_generator = <torch._C.Generator object>
#torch.bernoulli(input, out=None) -> Tensor
a = torch.Tensor(3, 3).uniform_(0, 1) # generate a uniform random matrix with range [0, 1]
print(a)
b = torch.bernoulli(a)
print(b)

a = torch.ones(3, 3) # probability of drawing "1" is 1
b = torch.bernoulli(a)
print(b)

a = torch.zeros(3, 3) # probability of drawing "1" is 0
b = torch.bernoulli(a)
print(b)

# torch.multinomial(input, num_samples, replacement=False, out=None) -> LongTensor

# torch.normal(means, std, out=None)
# torch.normal(mean=0.0, std, out=None)
# torch.normal(means, std=1.0, out=None)

#torch.rand(*sizes, out=None) -> Tensor
#torch.randn(*sizes, out=None) -> Tensor
#torch.randperm(n, out=None) -> LongTensor


#In-place random sampling
#torch.Tensor.bernoulli_() - in-place version of torch.bernoulli()
#torch.Tensor.cauchy_() - numbers drawn from the Cauchy distribution
#torch.Tensor.exponential_() - numbers drawn from the exponential distribution
#torch.Tensor.geometric_() - elements drawn from the geometric distribution
#torch.Tensor.log_normal_() - samples from the log-normal distribution
#torch.Tensor.normal_() - in-place version of torch.normal()
#torch.Tensor.random_() - numbers sampled from the discrete uniform distribution
#torch.Tensor.uniform_() - numbers sampled from the uniform distribution
