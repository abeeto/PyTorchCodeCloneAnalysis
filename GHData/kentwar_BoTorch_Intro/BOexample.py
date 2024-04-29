#%%
import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP

## Our fitness function
from botorch.test_functions import Hartmann 
from gpytorch.mlls import ExactMarginalLogLikelihood

neg_hartmann6 = Hartmann(dim=6, negate=True)

#%%
## First, we generate some random data and fit a SingleTaskGP for
## a 6-dimensional synthetic test function 'Hartmann6'.
## it has 6 local minima

train_x = torch.rand( 10 , 6 )
train_obj = neg_hartmann6( train_x ).unsqueeze( -1 )
model = SingleTaskGP( train_X = train_x , train_Y = train_obj )
mll = ExactMarginalLogLikelihood( model.likelihood , model )
fit_gpytorch_model( mll );

#%%
## Get our acquisition function

from botorch.acquisition import ExpectedImprovement

best_value = train_obj.max()
EI = ExpectedImprovement( model = model , best_f = best_value )

#%%
## Next, we optimize the analytic EI acquisition function using 50 
## random restarts chosen from 100 initial raw samples.

from botorch.optim import optimize_acqf

new_point_analytic, _ = optimize_acqf(
    acq_function=EI,
    bounds=torch.tensor([[0.0] * 6, [1.0] * 6]),
    q=1,
    num_restarts=20,
    raw_samples=100,
    options={},
)

#%%
## What is the solution after 100 steps?

print( new_point_analytic )
# %%
## Now, let's swap out the analytic acquisition function and replace it with an 
## MC version. Note that we are in the q = 1 case; for q > 1, an analytic 
## version does not exist.

from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler


sampler = SobolQMCNormalSampler( num_samples = 500 , seed = 0 , resample = False)        
MC_EI = qExpectedImprovement( model, best_f = best_value , sampler = sampler)

torch.manual_seed( seed = 0 ) # to keep the restart conditions the same
new_point_mc, _ = optimize_acqf(
    acq_function=MC_EI,
    bounds=torch.tensor([[0.0] * 6, [1.0] * 6]),
    q=1,
    num_restarts=20,
    raw_samples=100,
    options={},
)
# %%
new_point_mc
# %%
