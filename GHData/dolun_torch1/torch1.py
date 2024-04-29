import torch as tc
import torch.distributions as D
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import os

# https://pyro.ai/examples/normalizing_flows_i.html
# http://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline_coupling.html#spline_coupling
# http://docs.pyro.ai/en/stable/distributions.html#transforms

# tc.manual_seed(20)
K = 20
mix = D.Categorical(tc.ones(K,))
comp = D.Normal(tc.randn(K,), tc.rand(K,)*.2+.05)
gmm = D.MixtureSameFamily(mix, comp)

data = gmm.sample([1000])

print(-gmm.log_prob(data).mean())

base_dist = dist.Normal(tc.zeros(1), tc.ones(1))
spline_transform = T.Spline(1, count_bins=40, bound=4., order='quadratic')
flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])

# print(samples)
sns.histplot(data.numpy(), kde=False, bins=150, stat="density", alpha=.3)

tx = tc.linspace(-3, 3, steps=200)
pl.plot(tx, pl.exp(gmm.log_prob(tx)), color="red", lw=.8)

steps = 1000
optimizer = tc.optim.Adam(spline_transform.parameters(), lr=1e-2)

dataset = data.view(-1, 1)

for _ in range(10):
    weights_bs = D.Exponential(tc.ones_like(dataset)).sample()

    for step in range(steps):
        optimizer.zero_grad()
        loss = -(flow_dist.log_prob(dataset) * weights_bs).mean()
        loss.backward()
        optimizer.step()
        flow_dist.clear_cache()

        if step % 200 == 0:
            print(f'step: {step}, loss: {loss.item()}')

    # print(flow_dist.log_prob(tx.view(-1,1)))
    print("- "*15)
    res = pl.exp(flow_dist.log_prob(tx.view(-1, 1)).detach().view(-1))
    pl.plot(tx.numpy(), res, color="purple", ls="-", lw=.8, alpha=.2)
pl.show()
