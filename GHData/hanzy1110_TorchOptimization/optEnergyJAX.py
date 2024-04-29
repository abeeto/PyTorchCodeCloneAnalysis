#!/usr/bin/env python
from __future__ import division
import os
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

CORES = 4
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={CORES}'

import jax
from jax import jit, pmap
import jax.numpy as jnp
import optax

# f0 = 1e12

p, dim, N, iterations, max_iters = 3.0, 3, 6, 1000, 100

def pframe(X, p, dim):
    N = X.size // dim
    norms = jnp.sqrt(jnp.sum(X.reshape((-1, dim))**2, 1))
    allsums = jnp.matmul(X.reshape((-1, dim)), X.reshape((-1, dim)).T)
    energy = jnp.abs(allsums / (norms[:, None] * norms[None, :]))**p
    en = jnp.sum(energy) / N**2
    # for l in range(dim):
    #     M = p * energy * (X.reshape((-1, dim))[:, l][None, :]/allsums -
    #                 X.reshape((-1, dim))[:, l][:, None] / norms[:, None]**2)
    #     grad.reshape((-1, dim))[:, l] = np.sum(M, 1) * 2 / N**2.
    return en

def trainStep(u, p, dim, opt):
    x = opt.optimize(u)

    minf = opt.last_optimum_value()
    lastOptim = opt.last_optimize_result()
    return x, minf, lastOptim

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    f0 = 1e12
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (N,dim))
    # u = jnp.ones((N,dim))
    u = (u / np.sqrt(np.sum(u**2, 1))[:, None]).ravel()
    opt = optax.adam(learning_rate=0.1)
    params = {'u':u}
    opt_state = opt.init(params)

    for I in range(iterations):

        compute_loss = jit(lambda params: pframe(params['u'], p, dim))
        gradFun = jax.grad(compute_loss)
        grads = jax.grad(compute_loss)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        minf = compute_loss(params)
        err = jnp.abs(f0-minf)/jnp.abs(minf)

        if jnp.isclose(err, b=0, rtol=1e-6):
            relDist = jnp.linalg.norm(u-grads['u'])/jnp.linalg.norm(grads['u'])
            print('u', u)
            final_grad = gradFun(params)
            print('grad norm', jnp.linalg.norm(final_grad['u']))
            print('Solution', grads['u'])
            print('Relative distance-->', relDist)
            print('grads--->', grads)
            print('minf--->', minf)
            return
        if minf < f0:
            f0 = minf
            # print("optimum at ", x.T)
            # print("minimum value = ", minf)
            # print("result code = ", lastOptim)
            # X = x.reshape((-1, dim))
            # X = X / np.sqrt(np.sum(X**2, 1))[:, None]
            # fname = 'out/P_' + str(p) + '_dim_' + str(dim)+'_N_' + str(N)+'.out'
            # np.savetxt(fname, X, fmt='%.18f', delimiter='\t')

if __name__ == "__main__":
    main()
