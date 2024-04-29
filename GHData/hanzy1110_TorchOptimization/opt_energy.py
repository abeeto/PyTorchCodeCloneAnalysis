#!/usr/bin/env python
from __future__ import division
import nlopt
import os as os
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from numba import jit, float64, uint8, prange

# f0 = 1e12


def get_options(parser):
    """ Define command line options."""
    parser.add_option(
        "-p", dest="p", default=3.0,
        help="Power of the frame potential. Default: 3.0.")
    parser.add_option(
        "-d",
        "--dim",
        dest="dim",
        default=3,
        help="Dimension of the ambient space. Default: 3.")
    parser.add_option(
        "-N",
        dest="N",
        default=12,
        help="Number of particles. Default: 12.")
    parser.add_option(
        "-c",
        "--config",
        dest="conf",
        default=None,
        help="A starting configuration. Must be a string with the file name\
        relative to the script's directory. Default: None.")
    parser.add_option(
        "-i",
        "--iter",
        dest="iterations",
        default=100,
        help="A number of iterations to perform. Always 1 if a configuration\
        is given. Default: 100.")
    options, args = parser.parse_args()
    return float(options.p), int(options.dim), int(options.N),\
        int(options.iterations), options.conf


# @jit(float64(float64[:], float64[:], float64, uint8), nogil=True, cache=True)
def pframe(X, grad, p, dim):
    N = X.size // dim
    norms = np.sqrt(np.sum(X.reshape((-1, dim))**2, 1))
    allsums = np.matmul(X.reshape((-1, dim)), X.reshape((-1, dim)).T)
    energy = np.abs(allsums / (norms[:, None] * norms[None, :]))**p
    en = np.sum(energy) / N**2
    for l in range(dim):
        M = p * energy * (X.reshape((-1, dim))[:, l][None, :]/allsums -
                    X.reshape((-1, dim))[:, l][:, None] / norms[:, None]**2)
        grad.reshape((-1, dim))[:, l] = np.sum(M, 1) * 2 / N**2.
    return en


def trainStep(u, p, dim):

    opt = nlopt.opt(nlopt.LD_LBFGS, np.size(u))
    opt.set_min_objective(lambda x, v: pframe(x, v, p, dim))
    opt.set_ftol_rel(1e-18)

    x = opt.optimize(u)

    minf = opt.last_optimum_value()
    lastOptim = opt.last_optimize_result()
    return x, minf, lastOptim

def main():
    f0 = 1e12
    parser = OptionParser()
    p, dim, N, iterations, conf = get_options(parser)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    for I in range(iterations):
        if conf is not None:
            u = np.loadtxt(conf, delimiter='\t')
        else:
            u = np.random.randn(N, dim)
            u = (u / np.sqrt(np.sum(u**2, 1))[:, None]).ravel()

        x, minf, lastOptim = trainStep(u,p, dim)

        if minf < f0:
            f0 = minf
            # print("optimum at ", x.T)
            print("minimum value = ", minf)
            print("result code = ", lastOptim)
            X = x.reshape((-1, dim))
            X = X / np.sqrt(np.sum(X**2, 1))[:, None]
            fname = 'out/P_' + str(p) + '_dim_' + str(dim)+'_N_' + str(N)+'.out'
            np.savetxt(fname, X, fmt='%.18f', delimiter='\t')

if __name__ == "__main__":
    main()
