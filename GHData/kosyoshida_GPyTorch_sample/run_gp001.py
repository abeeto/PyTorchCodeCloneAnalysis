import torch
import gpytorch
import matplotlib.pylab as plt

from utils.functions import myfunc001
from utils.GPTrainer import Trainer
from utils.models import SingletaskGPModel
from utils.AcquisitionFunction import ExpectedImprovement

# training data
train_inputs = torch.rand((10))
train_targets = myfunc001(train_inputs)

# define likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# define prior dist. w.r.t. hyper parameters
l_prior = gpytorch.priors.NormalPrior(loc=torch.tensor(1.0), scale=torch.tensor(10.0))
s_prior = gpytorch.priors.NormalPrior(loc=torch.tensor(1.0), scale=torch.tensor(10.0))

# define GP
gpr = SingletaskGPModel(train_inputs, train_targets, likelihood, lengthscale_prior=l_prior, outputscale_prior=s_prior)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)

# define optimizer
optimizer = torch.optim.RMSprop(params=gpr.parameters(), lr=1e-2)

# optimizer marginal likelihood (mll) w.r.t. hyper parameters
trainer = Trainer(gpr, likelihood, optimizer, mll)
trainer.update_hyperparameter(2000)

# test data
test_inputs = torch.linspace(0, 1, 100)

gpr.eval()
likelihood.eval()
with torch.no_grad():
    predicts = likelihood(gpr(test_inputs))
    predicts_mean = predicts.mean
    predicts_std = predicts.stddev

EI = ExpectedImprovement()
ei = EI.get(predicts_mean, predicts_std, torch.min(train_targets))

# plot
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(test_inputs.numpy(), predicts_mean.numpy())
plt.fill_between(test_inputs.numpy(),
                 predicts_mean.numpy() - 0.9*predicts_std.numpy(),
                 predicts_mean.numpy() + 0.9*predicts_std.numpy(), alpha=0.4)
plt.plot(train_inputs.numpy(), train_targets.numpy(), "ro")

plt.subplot(122)
plt.plot(test_inputs, ei)
plt.show()
