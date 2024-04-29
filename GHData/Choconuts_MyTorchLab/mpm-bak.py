#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : mpm.py
@Author: Chen Yanzhen
@Date  : 2020/9/22 21:35
@Desc  : 
"""

import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import itertools
import time


class Timer:

    def __init__(self):
        self.time = time.time()
        self.verbose = False

    def tick(self, msg=''):
        new_time = time.time()
        if self.verbose:
            print('[Time]', msg, f'({new_time - self.time})')
        self.time = new_time


def log(tensor: torch.Tensor, tag='tensor'):
    filters = ['stress', 'aff_dist_prod', 'affine'] # ['tmp_vs', 'weighted_vs', , 'weight', 'weights']
    if True and tag not in filters and tag != 'tensor':
        return
    if True:
        print("[Tensor]", tag, tuple(tensor.shape), '{', torch.min(tensor).cpu().numpy(), '~',
              torch.mean(tensor).cpu().numpy(), '~', torch.max(tensor).cpu().numpy(),
              '>>', torch.sum(tensor).cpu().numpy(),
              '}')


def grid_iter(*shape):
    return itertools.product(*[range(dim) for dim in shape])


def eye(n=2):
    return torch.eye(n, device=device)


def index_path_fill(index_paths: torch.Tensor, values: torch.Tensor, target: torch.Tensor):
    '''

    :param index_paths: N * d
    :param values: N [* v]
    :param target: (..., d dims) [* v]
    :return:
    '''
    assert target.shape.__len__() == index_paths.shape[1] + 1 or target.shape.__len__() == index_paths.shape[1]
    dims = target.shape[:index_paths.shape[1]]
    target.reshape(-1, 1 if len(target.shape) == len(dims) else target.shape[-1])       # D * v
    target.index_put_()


""" Configurations """
quality = 2
device = 'cpu'
n_particles = 9000 // 25 * quality ** 2
n_grids = 25 * quality

dx, inv_dx = 1 / n_grids, float(n_grids)
dt = 5e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
x = torch.tensor([[np.random.rand() * 0.2 + 0.3 + 0.10 * (i // (n_particles // 3)),
                np.random.rand() * 0.2 + 0.05 + 0.32 * (i // (n_particles // 3))]
                for i in range(n_particles)], dtype=torch.float32).to(device).requires_grad_(False)  # N, 2
# x = torch.tensor([[(i // 46) / 100 + 0.25, (i - (i // 46) * 46) / 100 + 0.25]
#                 for i in range(n_particles)], dtype=torch.float32).to(device).requires_grad_(False)  # N, 2
v = torch.zeros((n_particles, 2), dtype=torch.float32).to(device).requires_grad_(False)              # N, 2
C = torch.zeros((n_particles, 2, 2), dtype=torch.float32).to(device).requires_grad_(False)           # N, 2, 2
F = torch.stack([torch.eye(2) for i in range(x.size(0))]).to(device).requires_grad_(False)           # N, 2, 2
Jp = torch.ones(n_particles, dtype=torch.float32).to(device).requires_grad_(False)                   # plastic deformation

grid_v = torch.zeros((n_grids, n_grids, x.size(1))).to(device).requires_grad_(False)  # grid node momentum/velocity = G, G, 2
grid_m = torch.zeros((n_grids, n_grids)).to(device).requires_grad_(False)  # grid node mass = G, G
boundary = torch.zeros((n_grids, n_grids, x.size(1))).requires_grad_(False)

for i, j in grid_iter(n_grids, n_grids):
    width = 3
    boundary[i, j] = torch.tensor([-1 if i < width else (1 if i >= n_grids - width else 0),
                      -1 if j < width else (1 if j >= n_grids - width else 0)])
    boundary = boundary.to(device)


U = torch.zeros(n_particles, 2, 2).to(device).requires_grad_(False)
sig = torch.zeros(n_particles, 2).to(device).requires_grad_(False)
V = torch.zeros(n_particles, 2, 2).to(device).requires_grad_(False)


def step():
    timer = Timer()
    global F, Jp, x, v, C, grid_v, grid_m, sig, U, V
    grid_v.fill_(0)
    grid_m.fill_(0)

    base = (x * inv_dx - 0.5).int()         # N, 2 (int)
    diff = x * inv_dx - base.float()        # N, 2
    log(diff, 'diff')
    weights = [0.5 * (1.5 - diff) ** 2,
               0.75 - (diff - 1) ** 2,
               0.5 * (diff - 0.5) ** 2]     # 3, N, 2
    # log(torch.stack(weights), 'weights')
    F = (eye() + dt * C) @ F            # deformation gradient
    h = torch.exp(10 * (1.0 - Jp))      # Hardening coefficient: snow gets harder when compressed
    la = lambda_0 * h * 1
    mu = 0.0
    timer.tick()
    # SIG 可能不一样
    log(F, 'F')
    U, sig, V = torch.svd(F)
    log(sig, 'sig')
    # log(F - U @ eye() * sig.unsqueeze(2) @ V)
    log(U @ V.transpose(1, 2), 'UV')
    timer.tick('svd')
    J = torch.ones(n_particles).to(device)

    J *= sig[:, 0]
    J *= sig[:, 1]
    J = J.reshape(-1, 1, 1)
    log(J, 'J')
    F = eye() * J ** 0.5

    # stresses and affine

    stress = 2 * mu * (F - U @ V.transpose(1, 2)) @ F.transpose(1, 2) + eye() * la.reshape(-1, 1, 1) * J * (J - 1)
    # stress = eye() * la.reshape(-1, 1, 1) * J * (J - 1)
    log(J * (J - 1), 'stress')
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress

    affine = stress + p_mass * C
    log(C)
    log(affine, 'affine')

    # offset: N, 9, 2
    offset = torch.stack([torch.tensor([i, j]).to(device).int() for i, j in grid_iter(3, 3)], 0)
    dists = (offset - diff.unsqueeze(1)) * dx
    weight = torch.stack([weights[i][:, 0] * weights[j][:, 1] for i, j in grid_iter(3, 3)], 1)  # N,9
    affine_batches = affine.unsqueeze(1).expand(-1, 9, -1, -1).reshape(-1, 2, 2)
    aff_dist_prod = affine_batches @ dists.reshape(-1, 2, 1)
    log(affine_batches)
    log(dists.reshape(-1, 2, 1))
    log(aff_dist_prod, 'aff_dist_prod')
    tmp_vs = p_mass * v.unsqueeze(1) + aff_dist_prod.reshape(n_particles, 9, -1)    # N,9,2
    log(weight, 'weight')
    log(tmp_vs, 'tmp_vs')
    # log(tmp_vs / p_mass)
    weighted_vs = weight.unsqueeze(2) * tmp_vs
    weighted_ms = weight * p_mass
    log(weighted_vs, 'weighted_vs')
    # log(weighted_vs / p_mass)

    # weighted_v0 = weight.unsqueeze(2) * v.unsqueeze(1)
    # weighted_m0 = weight.unsqueeze(2) * p_mass

    # for p in range(n_particles):
    #     # F[p] = (eye() + dt * C[p]) @ F[p]   # deformation gradient update
    #     # h = torch.exp(10 * (1.0 - Jp[p]))   # Hardening coefficient: snow gets harder when compressed
    #     # la = lambda_0 * h
    #     # mu = 0.0
    #     # U, sig, V = torch.svd(F[p])
    #     # J = 1.0
    #     # stress = 2 * mu * (F[p] - U @ V.t()) @ F[p].t() + eye() * la * J * (J - 1)
    #     # # stress = 2 * mu * (F[p] - U[p] @ V[p].t()) @ F[p].t() + eye2() * la[p] * J[p] * (J[p] - 1)
    #     # stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    #     # affine = stress + p_mass * C[p]
    #     # for i, j in grid_iter(3, 3): # Loop over 3x3 grid node neighborhood
    #     #     off = torch.tensor([i, j])
    #     #     dpos = (off.float() - diff[p]) * dx
    #     #     w = weights[i][p][0] * weights[j][p][1]
    #     #     grid_v[base[p] + off] += w * (p_mass * v[p] + affine @ dpos)
    #     #     grid_m[base[p] + off] += w * p_mass
    #     aff = affine[p]
    #     for i, j in grid_iter(3, 3): # Loop over 3x3 grid node neighborhood
    #         off = torch.tensor([i, j])
    #         dpos = (off.float() - diff[p]) * dx
    #         w = weights[i][p][0] * weights[j][p][1]
    #         grid_v[base[p] + off] += w * (p_mass * v[p] + aff @ dpos)
    #         grid_m[base[p] + off] += w * p_mass

    tmp = (base.unsqueeze(1) + offset.unsqueeze(0)).reshape(n_particles * 9, 2).long()
    x_idx, y_idx = tmp[:, 0], tmp[:, 1]     # N * 9
    grid_v.index_put_(indices=[x_idx, y_idx], values=weighted_vs.reshape(n_particles * 9, 2), accumulate=True)
    grid_m.index_put_(indices=[x_idx, y_idx], values=weighted_ms.reshape(n_particles * 9), accumulate=True)
    timer.tick('p2g')

    # f + mg = ma
    non_zero_grid_m = torch.where(grid_m > 1e-6, grid_m, torch.ones_like(grid_m) * p_mass)
    log(grid_v, 'grid_v 1')
    grid_v *= 1 / non_zero_grid_m.unsqueeze(-1)
    log(grid_v, 'grid_v 2')
    grid_v[:, :, 1] -= dt * 50
    log(grid_v, 'grid_v')
    log(grid_m, 'grid_m')

    vel_bound_dot = torch.sign(grid_v) * boundary
    grid_v = torch.where(vel_bound_dot > 0, torch.zeros_like(grid_v), grid_v)
    log(grid_v, 'grid_v_clipped')
    # for i, j in grid_iter(*grid_m.shape[:2]):
    #     if grid_m[i, j] > 0: # No need for epsilon here
    #         # grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]    # Momentum to velocity
    #         # grid_v[i, j][1] -= dt * 50  # gravity
    #
    #         # boundary condition
    #         if i < 3 and grid_v[i, j][0] < 0 or i > n_grids - 3 and grid_v[i, j][0] > 0:
    #             grid_v[i, j][0] = 0
    #         if j < 3 and grid_v[i, j][1] < 0 or j > n_grids - 3 and grid_v[i, j][1] > 0:
    #             grid_v[i, j][1] = 0

    # G2P
    # 9, N, 2
    selected = grid_v[x_idx, y_idx].reshape(n_particles, 9, -1)
    weighted_sel = weight.unsqueeze(2) * selected
    v = torch.sum(weighted_sel, 1)

    # directly on V
    v[:, 1] -= dt * 50
    # v = torch.where(x <= 2 * dx, torch.max(v, torch.zeros_like(v)), v)
    # v = torch.where(x >= 1 - 2 * dx, torch.min(v, torch.zeros_like(v)), v)
    out = weighted_sel.reshape(-1, 9, 2, 1) * dists.reshape(-1, 9, 1, 2)
    C = 4 * inv_dx * inv_dx * torch.sum(out, 1)

    x += dt * v
    # x = (base.float() + offset.float()[4]) * dx        # grid magnetic

    # for p in range(x.size(0)):      # grid to particle (G2P)
    #     new_v = torch.zeros(2).to(device)
    #     new_C = torch.zeros(2, 2).to(device)
    #     for i, j in grid_iter(3, 3):    # loop over 3x3 grid node neighborhood
    #         offset = torch.tensor([i, j]).to(device)
    #         dpos = offset.float() - diff[p]
    #         g_v = grid_v[tuple(base[p] + offset)]
    #         weight = weights[i][p][0] * weights[j][p][1]
    #         new_v += weight * g_v
    #         # new_C += 4 * inv_dx * weight * torch.tensor([
    #         #     [g_v[0] * dpos[0], g_v[0] * dpos[1]],
    #         #     [g_v[1] * dpos[0], g_v[1] * dpos[1]]
    #         # ]).to(device)
    #         new_C += 4 * inv_dx * weighted_sel[p, i * 3 + j].reshape(2, 1) * inv_dx * dists[p, i * 3 + j].reshape(1, 2)
    #         # v[p] = new_C
    #         C[p] = new_C
    #         x[p] += dt * v[p]   # advection

    log(x, 'x')
    log(v, 'v')
    log(C, 'C')
    timer.tick('g2p')



def update_points(num):
    timer = Timer()
    step()
    x_cpu = x
    point_ani.set_data(x_cpu[:,0], x_cpu[:,1])
    # point_ani.set_data(indices[:, 0] * dx, indices[:, 1] * dx)
    timer.tick('frame')
    return point_ani,


fig = plt.figure(tight_layout=True)
plt.xlim(0, 1)
plt.ylim(0, 1)

indices = torch.tensor([(i, j) for i in range(n_grids) for j in range(n_grids)]).to(device)
values = grid_m[indices[:, 0], indices[:, 1]]
indices = indices.where(values.unsqueeze(1) > 1e-6, -torch.ones_like(indices))

x_cpu = x.cpu()
point_ani, = plt.plot(x_cpu[:,0], x_cpu[:,1], '.')
plt.grid(ls="--")
ani = animation.FuncAnimation(fig, update_points, np.arange(0, 100), interval=1, blit=True)
plt.show()

if __name__ == '__main__':
 pass