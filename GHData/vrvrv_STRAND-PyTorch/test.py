import torch
import pickle
import numpy as np
from src.strand.functions import *

import argparse

rank = 16
data = 'ts'
model = 'strand'


def Chat_str(ckpt_dir, Y):
    ckpt = torch.load(ckpt_dir, map_location=torch.device('cpu'))

    y_tr = Y.sum(axis=(2, 3, 4, -2))

    __m00 = y_tr[:2, :2].sum(axis=(0, 1)).float() / y_tr.sum(dim=(0, 1))
    __m01 = y_tr[:2, 2].sum(axis=(0)).float() / y_tr.sum(dim=(0, 1))
    __m10 = y_tr[2, :2].sum(axis=(0)).float() / y_tr.sum(dim=(0, 1))
    __m11 = y_tr[2, 2].float() / y_tr.sum(dim=(0, 1))

    missing_rate = torch.stack([__m00, __m01, __m10, __m11]).reshape(2, 2, -1)

    T = stack(
        _T0=ckpt['state_dict']['_T0'],
        _t=ckpt['state_dict']['_t'],
        _r=ckpt['state_dict']['_r'],
        e_dim=ckpt['hyper_parameters']['e_dim'],
        n_dim=ckpt['hyper_parameters']['n_dim'],
        c_dim=ckpt['hyper_parameters']['c_dim'],
        rank=ckpt['hyper_parameters']['rank']
    )

    F = factors_to_F(
        _t=ckpt['state_dict']['_t'],
        _r=ckpt['state_dict']['_r'],
        _e=ckpt['state_dict']['_e'],
        _n=ckpt['state_dict']['_n'],
        _c=ckpt['state_dict']['_c'],
        e_dim=ckpt['hyper_parameters']['e_dim'],
        n_dim=ckpt['hyper_parameters']['n_dim'],
        c_dim=ckpt['hyper_parameters']['c_dim'],
        missing_rate=missing_rate,
        rank=ckpt['hyper_parameters']['rank'],
        uniform_missing_rate=True
    )

    tf = T * F

    chat = (
        tf.matmul(logit_to_distribution(ckpt['state_dict']['lamb']).float()) \
        * Y.sum(dim=(0, 1, 2, 3, 4, 5))
    ).squeeze()

    phi = Phi(T=T, F=F, lambda_or_Lambda=('lambda', ckpt['state_dict']['lamb']))

    bias_correction = (tf * phi.mean(dim=-3, keepdim=True)).sum(-1)

    chat_bias_corrected = chat * bias_correction.transpose(-1, -2)

    return chat, chat_bias_corrected


def Chat_ts(ckpt_dir):
    ckpt = torch.load(ckpt_dir, map_location=torch.device('cpu'))

    S0 = ckpt['state_dict']['S0']
    E0 = ckpt['state_dict']['E0']
    a0 = ckpt['state_dict']['a0']
    b0 = ckpt['state_dict']['b0']
    k0 = ckpt['state_dict']['k0']
    k1 = ckpt['state_dict']['k1']
    k2 = ckpt['state_dict']['k2']

    S0s = torch.softmax(
        torch.cat([S0, torch.zeros([2, 2, 1, ckpt['hyper_parameters']['rank']])], dim=2), dim=2
    )

    S1 = torch.stack([
        S0s[0, 0, :, :], S0s[1, 0, :, :], 0.5 * S0s[:, 0, :, :].sum(0),
        S0s[1, 1, :, :], S0s[0, 1, :, :], 0.5 * S0s[:, 1, :, :].sum(0),
                                          0.5 * (S0s[0, 0, :, :] + S0s[1, 1, :, :]),
                                          0.5 * (S0s[1, 0, :, :] + S0s[0, 1, :, :]),
                                          0.25 * S0s.sum(dim=(0, 1))
    ]).reshape(3, 3, 1, -1, ckpt['hyper_parameters']['rank'])

    idx = torch.arange(Y.size(-1))

    E = torch.exp(E0[..., idx])

    card = list(ckpt['hyper_parameters']['snv_shape'])[2: -2]
    card_prod = np.prod(card)

    I = np.arange(card_prod)
    card = card

    card = np.array(card, copy=False)
    C = card.flatten()
    A = np.mod(
        np.floor(
            np.tile(I.flatten().T, (len(card), 1)).T /
            np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))),
                    (len(I), 1))),
        np.tile(C[::-1], (len(I), 1)))

    idex = A[:, ::-1]

    a1 = torch.exp(
        torch.cat([a0, a0, torch.zeros((2, ckpt['hyper_parameters']['rank']))], dim=0)
    ).reshape(3, 2, ckpt['hyper_parameters']['rank'])

    A = (a1[:, 0, :][:, None, :] * a1[:, 1, :][None, :, :]).reshape(
        3, 3, 1, 1, ckpt['hyper_parameters']['rank']
    )

    B = torch.exp(
        torch.stack([
            b0[0, :] + b0[1, :], b0[0, :] - b0[1, :], b0[0, :],
            b0[1, :] - b0[0, :], -b0[1, :] - b0[0, :], -b0[0, :],
            b0[1, :], -b0[1, :], torch.zeros(b0[0, :].shape)
        ]).reshape(3, 3, 1, 1, ckpt['hyper_parameters']['rank'])
    )

    _cbiases = {}

    _cbiases[0] = torch.cat(
        [torch.zeros([1, ckpt['hyper_parameters']['rank']]), k0], dim=0
    )
    _cbiases[1] = torch.cat(
        [torch.zeros([1, ckpt['hyper_parameters']['rank']]), k1], dim=0
    )
    _cbiases[2] = torch.cat(
        [torch.zeros([1, ckpt['hyper_parameters']['rank']]), k2], dim=0
    )

    final_tensor = []
    for r in range(idex.shape[0]):
        current_term = []
        for c in range(idex.shape[1]):
            current_term.append(
                _cbiases[c][idex[r, c].astype(int), :]
            )
        final_tensor.append(
            torch.stack(current_term).sum(dim=0)
        )

    K = torch.exp(
        torch.stack(final_tensor).reshape(1, 1, -1, 1, ckpt['hyper_parameters']['rank'])
    )

    S = S1 * A * B * K

    return torch.matmul(
        S.reshape(-1, ckpt['hyper_parameters']['rank']), E
    ).reshape(3, 3, -1, 96, len(idx))

def main(args):

    with open(f"data/{args.data}/snv.pkl", "rb") as f:
        Y = torch.from_numpy(pickle.load(f))

    chat_str, chat_str_bias_corrected = Chat_str(args.str_ckpt, Y)
    chat_ts = Chat_ts(args.ts_ckpt)

    Y = Y.reshape(-1, 96)

    idx = Y.sum(-1) != 0

    p = Y[idx] / Y[idx].sum(dim=(0, 1, 2, 3))
    p_cond = Y[idx] / Y[idx].sum(dim=-1, keepdim=True)

    chat_ts_flatten = chat_ts.reshape(-1, 96)

    # p(l, v | d)
    phat_ts = chat_ts / chat_ts.sum(dim=(0, 1, 2, 3))
    phat_str = chat_str.reshape(3, 3, -1, Y.size(-2), Y.size(-1))
    phat_str_bc = (
        chat_str_bias_corrected / chat_str_bias_corrected.sum(dim=-2, keepdim=True)
    ).reshape(3, 3, -1, Y.size(-2), Y.size(-1))

    phat_ts_cond = chat_ts_flatten[idx] / chat_ts_flatten[idx].sum(dim=-1, keepdim=True)

    phat_str_flatten = phat_str.reshape(-1, 96)
    phat_str_cond = phat_str_flatten[idx] / phat_str_flatten[idx].sum(dim=-1, keepdim=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data prepare')

    parser.add_argument('--data', type=str, default='ts')
    parser.add_argument('--ts_ckpt', type=str)
    parser.add_argument('--str_ckpt', type=str)


