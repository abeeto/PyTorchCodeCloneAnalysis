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
import jaxopt

# f0 = 1e12

p, dim, N, iterations, max_iters = 3.0,3, 6, 1000, 100

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

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    f0 = 1e12
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (N,dim))
    # u = jnp.ones((N,dim))
    u = (u / np.sqrt(np.sum(u**2, 1))[:, None]).ravel()
    params = {'u':u}
    # for I in range(iterations):
    compute_loss = jit(lambda params: pframe(params['u'], p, dim))
    loss_grad = jax.grad(compute_loss)
    # compute_loss = lambda params: pframe(params['u'], p, dim)
    solver = jaxopt.LBFGS(fun=compute_loss,
                          maxls=1000, use_gamma=False, jit=True,
                          increase_factor=10,
                          decrease_factor=0.1,
                          maxiter=10000, tol=1e-6)
    init_params = solver.init_state(params)
    sol, state = solver.run(params)
    minf = compute_loss(params)
    final_grad = loss_grad(params)
    if minf < f0:
        f0 = minf
        # print("optimum at ", x.T)
        print("minimum value = ", minf)
        print('solver output-->',  sol['u'])
        print('initial point-->', u)
        print('Relative Distance between solutions:')
        print(jnp.linalg.norm(u-sol['u'])/jnp.linalg.norm(sol['u']))
        print('final grad--->',jnp.linalg.norm(final_grad['u']))
        # print("result code = ", lastOptim)
        # X = x.reshape((-1, dim))
        # X = X / np.sqrt(np.sum(X**2, 1))[:, None]
        # fname = 'out/P_' + str(p) + '_dim_' + str(dim)+'_N_' + str(N)+'.out'
        # np.savetxt(fname, X, fmt='%.18f', delimiter='\t')

if __name__ == "__main__":
    main()
