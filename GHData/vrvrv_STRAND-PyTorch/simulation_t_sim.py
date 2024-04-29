import os
import sys
import json
import h5py
import torch
import pickle
import argparse
from sklearn.decomposition import NMF

# from src.strand.functions import *
from itertools import chain
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from statistics import mean

cs = torch.nn.CosineSimilarity(dim=0)

r_range = [5]
n_range = [10, 100, 1000, 3000]
m_range = [10, 100, 1000, 10000]


def logit(tensor: torch.Tensor, eps=1e-20) -> torch.Tensor:
    """ Logit transformation

    Args:
        tensor (`torch.Tensor`): The input tensor.
        eps (: obj: `float`): The small value for numerical stability.

    Returns :
        logit(tensor)

    """
    denom = tensor[-1]
    denom[denom < eps] = eps

    odd_ratio = tensor[:-1] / denom
    odd_ratio[odd_ratio < eps] = eps

    if isinstance(odd_ratio, torch.Tensor):
        logit = torch.log(odd_ratio)
    elif isinstance(odd_ratio, np.ndarray):
        logit = torch.log(torch.from_numpy(odd_ratio).float())
    else:
        raise TypeError
    return logit


def logit_to_distribution(tensor: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.cat([tensor, torch.zeros((1, tensor.size(dim=1)), device=tensor.device)], dim=0), dim=0)


def stack(_T0: torch.Tensor, _t: torch.Tensor, _r: torch.Tensor, e_dim: int, n_dim: int, c_dim: int):
    [[_cl, _cg], [_tl, _tg]] = _T0

    cl = logit_to_distribution(_cl)
    cg = logit_to_distribution(_cg)
    tl = logit_to_distribution(_tl)
    tg = logit_to_distribution(_tg)

    t = logit_to_distribution(_t)
    r = logit_to_distribution(_r)

    V, K = _T0.size()[-2:]
    T = torch.empty((3, 3, e_dim, n_dim, c_dim, V + 1, K), device=_T0.device)

    t0 = t[0]
    r0 = r[0]

    c_ = r0 * cl + (1 - r0) * cg
    t_ = r0 * tl + (1 - r0) * tg
    _l = t0 * cl + (1 - t0) * tl
    _g = t0 * cg + (1 - t0) * tg

    __ = t0 * r0 * cl + t0 * (1 - r0) * cg + (1 - t0) * r0 * tl + (1 - t0) * (1 - r0) * tg

    # T = torch.stack([
    #     cl, cg, c_, tl, tg, t_, _l, _g, __
    # ]).reshape(3, 3, 1, 1, 1, V+1, K).to(_T0.device).float()

    T[0, 0] = cl
    T[1, 0] = tl
    T[2, 0] = _l
    T[0, 1] = cg
    T[1, 1] = tg
    T[2, 1] = _g
    T[0, 2] = c_
    T[1, 2] = t_
    T[2, 2] = __

    return T


def factors_to_F(
        _t: torch.Tensor,
        _r: torch.Tensor,
        _e: torch.Tensor,
        _n: torch.Tensor,
        _c: torch.Tensor,
        _at: torch.Tensor,
        _ar: torch.Tensor,
        rank: int,
        missing_rate: torch.Tensor = None,
        reduction: bool = True
) -> torch.Tensor:
    e = logit_to_distribution(_e)
    n = logit_to_distribution(_n)
    c = logit_to_distribution(_c)

    e_dim = e.size(0)
    n_dim = n.size(0)
    c_dim = c.size(0)

    if reduction:
        F = torch.ones((3, 3, e_dim, n_dim, c_dim, 1, rank), device=_t.device)
    else:
        F = torch.ones((3, 3, e_dim, n_dim, c_dim, missing_rate.size(-1), rank), device=_t.device)
        # missing_rate = missing_rate.unsqueeze(-1)

    bt = torch.exp(_t)
    br = torch.exp(_r)
    at = torch.exp(_at)
    ar = torch.exp(_ar)

    F *= torch.stack([
        at * bt / (1 + at + bt + at * bt),
        at / (1 + at + bt + at * bt),
        1 / (1 + at)
    ]).reshape(3, 1, 1, 1, 1, 1, rank)

    F *= torch.stack([
        ar * br / (1 + ar + br + ar * br),
        ar / (1 + ar + br + ar * br),
        1 / (1 + ar)
    ]).reshape(1, 3, 1, 1, 1, 1, rank)

    for l in range(e_dim):
        F[:, :, l] *= e[l]
    for l in range(n_dim):
        F[:, :, :, l] *= n[l]
    for l in range(c_dim):
        F[:, :, :, :, l] *= c[l]

    return F


def stack_wo_missing(_T0: torch.Tensor, _t: torch.Tensor, _r: torch.Tensor, e_dim: int, n_dim: int, c_dim: int):
    [[_cl, _cg], [_tl, _tg]] = _T0

    cl = logit_to_distribution(_cl)
    cg = logit_to_distribution(_cg)
    tl = logit_to_distribution(_tl)
    tg = logit_to_distribution(_tg)

    V, K = _T0.size()[-2:]

    T = torch.stack([
        cl, cg, tl, tg
    ]).reshape(2, 2, 1, 1, 1, V + 1, K).to(_T0.device).float()

    return T


def factors_to_F_wo_missing(
        _t: torch.Tensor,
        _r: torch.Tensor,
        _e: torch.Tensor,
        _n: torch.Tensor,
        _c: torch.Tensor,
        _at: torch.Tensor,
        _ar: torch.Tensor,
        rank: int,
) -> torch.Tensor:
    t = logit_to_distribution(_t)
    r = logit_to_distribution(_r)
    e = logit_to_distribution(_e)
    n = logit_to_distribution(_n)
    c = logit_to_distribution(_c)

    t_dim = t.size(0)
    r_dim = r.size(0)
    e_dim = e.size(0)
    n_dim = n.size(0)
    c_dim = c.size(0)

    F = torch.ones((t_dim, r_dim, e_dim, n_dim, c_dim, 1, rank), device=_t.device)

    for l in range(t_dim):
        F[l] *= t[l]
    for l in range(r_dim):
        F[:, l] *= r[l]
    for l in range(e_dim):
        F[:, :, l] *= e[l]
    for l in range(n_dim):
        F[:, :, :, l] *= n[l]
    for l in range(c_dim):
        F[:, :, :, :, l] *= c[l]

    return F


def match_signatures(theta, theta_pred, rank):
    """
    Args :
        theta : K x D
        theta_pred : K x D
    Returns : perm
        ex. perm = [0,2,3,1,4]
        theta_pred [0] <-> theta [0]
        theta_pred [2] <-> theta [1]
    """

    theta = theta.float()
    theta_pred = theta_pred.float()

    cost = np.zeros((rank, rank))
    for a in range(rank):
        for b in range(rank):
            cost[a, b] = torch.linalg.norm(theta[a] - theta_pred[b], ord=1)

    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind


def load_result(rank, n, m, sid, i):
    with open(f'data/simulation_{sid}/rank_{rank}_m_{m}_n_{n}_param.pkl', 'rb') as f:
        param = pickle.load(f)

    try:
        pred = torch.load(
            f'checkpoints/simulation_{sid}/multiple_run/rank_{rank}_n_{n}_m_{m}_{i}.ckpt',
            map_location=torch.device('cpu')
        )['state_dict']
    except:
        return None, None

    T = stack(
        _T0=pred['_T0'],
        _t=pred['_t'],
        _r=pred['_r'],
        e_dim=12,
        n_dim=4,
        c_dim=2,
    ).float()

    F = factors_to_F(
        _t=pred['_t'],
        _r=pred['_r'],
        _e=pred['_e'],
        _n=pred['_n'],
        _c=pred['_c'],
        _at=pred['_at'],
        _ar=pred['_ar'],
        rank=rank
    ).float()

    tf = (T * F).sum((0, 1, 2, 3, 4))

    theta = logit_to_distribution(pred['lamb'])
    true_theta = param['theta']
    perm = match_signatures(true_theta.float(), theta.float(), rank=rank)

    tf = tf[:, perm]

    ## True signature

    [[cl_true, cg_true], [tl_true, tg_true]] = param['T0']

    _cl_true = logit(cl_true)
    _cg_true = logit(cg_true)
    _tl_true = logit(tl_true)
    _tg_true = logit(tg_true)

    _T0 = torch.stack([_cl_true, _cg_true, _tl_true, _tg_true]).reshape(2, 2, 95, -1)

    T = stack_wo_missing(
        _T0=_T0,
        _t=logit(param['factors']['t']),
        _r=logit(param['factors']['r']),
        e_dim=12,
        n_dim=4,
        c_dim=2,
    ).float()

    F = factors_to_F_wo_missing(
        _t=logit(param['factors']['t']),
        _r=logit(param['factors']['r']),
        _e=logit(param['factors']['e']),
        _n=logit(param['factors']['n']),
        _c=logit(param['factors']['c']),
        _at=logit(param['factors']['at']),
        _ar=logit(param['factors']['ar']),
        rank=rank
    ).float()

    tf_true = (T * F).sum((0, 1, 2, 3, 4))

    ## NMF signature

    with h5py.File(f'data/simulation_{sid}/rank_{rank}_m_{m}_n_{n}.hdf5', 'r') as f:
        snv = np.array(f['count_tensor'])

    nmf = NMF(rank, solver='mu', beta_loss='kullback-leibler', init='random', max_iter=500)

    y = snv.sum((0, 1, 2, 3, 4))
    tf_nmf = nmf.fit_transform(y)

    C = tf_nmf.sum(0)
    tf_nmf = tf_nmf / C
    theta_nmf = nmf.components_ / (nmf.components_.sum(0) +1e-5)

    perm = match_signatures(true_theta.float(), torch.from_numpy(theta_nmf).float(), rank=rank)
    tf_nmf = torch.from_numpy(tf_nmf)[:, perm].float()

    error = mean(cs(tf_true, tf).tolist())
    error_nmf = mean(cs(tf_true, tf_nmf).tolist())
    error = cs(tf_true, tf).tolist()
    error_nmf = cs(tf_true, tf_nmf).tolist()
    # error_nmf = float(torch.abs(tf_true- tf_nmf).mean())

    return error, error_nmf


def main(args):
    strand_res = dict()
    nmf_res = dict()

    with tqdm(total=len(n_range) * len(m_range), desc=f"Fig A (Simulation {args.sid})") as pbar:
        for row, n in enumerate(n_range):
            for col, m in enumerate(m_range):
                strand_res[f"n:{n},m:{m}"] = []
                nmf_res[f"n:{n},m:{m}"] = []

                for i in range(1, 11):
                    strand_sr, nmf_sr = load_result(args.rank, n, m, sid=args.sid, i=i)
                    strand_res[f"n:{n},m:{m}"].append(strand_sr)
                    nmf_res[f"n:{n},m:{m}"].append(nmf_sr)

                pbar.update(1)

    os.makedirs(f"assets/simulation_{args.sid}", exist_ok=True)

    # plt.savefig(f"assets/simulation_{args.sid}/fig_a.pdf")

    with open(f"assets/simulation_{args.sid}/signature_similarities.json", "w") as f:
        json.dump({'strand': strand_res, 'nmf': nmf_res}, f)

    print(f"saved at : assets/simulation_{args.sid}/signature_similarities.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw figure b')
    parser.add_argument('--sid', type=int, default=1)
    parser.add_argument('--rank', type=int, default=10)
    args = parser.parse_args()
    main(args)
