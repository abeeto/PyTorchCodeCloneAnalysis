import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KinematicRnnModel(nn.Module):
    def __init__(self, dt=0.01, seq_len=10, return_seq=False):
        super().__init__()
        # state [x, y, cos(theta), sin(theta), v, w]
        self.dt = dt
        self.seq_len = seq_len
        self.return_seq = return_seq

    def forward(self, u:torch.Tensor, h=None):
        assert u.ndimension() == 3 # (batch, seq, 2[a, alpha])
        if h is None:
            print('h is None')
            h = torch.tensor([0, 0, 1, 0, 0, 0]).double().repeat(u.size(0), 1)
        # unbind action in the time dimension
        if self.return_seq:
            x = []
        i = 0
        for a in torch.unbind(u, dim=1):
            i += 1
            cw = torch.cos(h[:, 5]*self.dt).double()
            sw = torch.sin(h[:, 5]*self.dt).double()
            dx = h[:, 4]*h[:, 2]*self.dt
            dy = h[:, 4]*h[:, 3]*self.dt
            dc = h[:, 2]*(cw - 1) - h[:, 3]*sw
            ds = h[:, 3]*(cw - 1) + h[:, 2]*sw
            dv = a[:, 0]*self.dt
            dw = a[:, 1]*self.dt
            h = h + torch.stack([dx, dy, dc, ds, dv, dw], dim=-1)
            if self.return_seq:
                x.append(h.detach())
        return torch.stack(x).permute(1, 0, 2) if self.return_seq else h


class FCNN(nn.Module):
    def __init__(self, n_layers, units, in_dim, op_dim,
        ac_fn=F.leaky_relu, op_act=F.leaky_relu):
        super().__init__()
        self.units = units if isinstance(units, list) else [units]*n_layers
        self.ac_fn = ac_fn if isinstance(ac_fn, list) else [ac_fn]*(n_layers-1)
        self.op_act = op_act
        self.layers = []
        self.layers.append(nn.Linear(in_dim, self.units[0]))
        for i in range(1, n_layers-1):
            self.layers.append(nn.Linear(self.units[i-1], self.units[i]))
        self.layers.append(nn.Linear(self.units[-1], op_dim))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for ac_fn, layer in zip(self.layers[:-1], self.ac_fn[:-1]):
            print(layer, ac_fn)
            x = ac_fn(layer(x))
            print(x)
        print(self.layers[-1], self.op_act)
        x = self.op_act(self.layers[-1](x))
        return x


class GRU_NN(FCNN):
    def __init__(self, *, in_dim, op_dim, n_layers, units, latent_size,
        num_rnn_layers=1, ac_fn=F.leaky_relu, op_act=F.leaky_relu):
        super().__init__(n_layers, units, latent_size, op_dim, ac_fn, op_act)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=latent_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x, h=None):
        op, hx = self.gru(x.double(), h)
        return super().forward(op), hx

class Policy(nn.Module):
    def __init__(self, input_dim, op_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, op_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

class MBRLPolicy(nn.Module):
    def __init__(self, seq_len, dt, ret_state=True, ip_dim=None, op_dim=None):
        super().__init__()
        self.dt = dt
        self.seq_len = seq_len
        self.ret_state = ret_state
        self.policy = Policy(ip_dim or 12, op_dim or 2).double()

    def forward(self, state, pstate, target):
        if pstate is None:
            pstate = state.clone()
            assert id(pstate) != id(state)

        x, y, c, s, v, w = state.clone()
        target = torch.cat([target, target])
        u_seq, states = [], []
        for i in range(self.seq_len):
            u = self.policy(torch.cat([pstate, state]) - target)*.5
            u_seq.append(u.clone())
            cw = torch.cos(w*self.dt).double()
            sw = torch.sin(w*self.dt).double()
            x = x + v*c*self.dt
            y = y + v*s*self.dt
            c = c*cw - s*sw
            s = s*cw + c*sw
            v = torch.clamp(v + u[0]*self.dt, -0.1, 0.3)
            w = torch.clamp(w + u[1]*self.dt, -np.pi/4, np.pi/4)
            pstate = state.clone()
            assert id(pstate) != id(state)
            state = torch.stack([x, y, c, s, v, w])
            states.append(state.clone())
        # print(u_seq)
        return u_seq, states


class Observe(nn.Module):
    def __init__(self):
        super().__init__()
        self.ZERO, self.ONE = torch.tensor([0, 1]).double()


    def forward(self, state, target):
        x, y, c, s, v, w = state
        tx, ty, tc, ts, _, _ = target
        d = torch.tensor([[x, y, self.ZERO]]).transpose(1, 0)
        r = torch.tensor([
                [c, s, self.ZERO],
                [-s, c, self.ZERO],
                [self.ZERO, self.ZERO, self.ONE]
            ])
        h = torch.cat([
                torch.cat([r, -r@d], dim=1),
                torch.tensor([[self.ZERO, self.ZERO, self.ZERO, self.ONE]])
            ])
        v1 = torch.tensor([[tx, ty, self.ZERO, self.ONE]]).transpose(1, 0)
        v2 = torch.tensor([
                [x + tc, y + ts, self.ZERO, self.ONE]
            ]).transpose(1, 0)
        return torch.cat([
            (h@v1).squeeze()[:-2], (h@v2).squeeze()[:-2], v[None], w[None]
        ])


class MBRLPolicy2(MBRLPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe = Observe()
        self.policy = Policy(12, 2).double()

    def forward(self, state, pstate, target):
        if pstate is None:
            pstate = state.clone()
            assert id(pstate) != id(state)

        x, y, c, s, v, w = state.clone()
        u_seq, states = [], []
        for i in range(self.seq_len):
            u = self.policy(torch.cat([
                self.observe(pstate, target),
                self.observe(state, target)
            ]))
            u_seq.append(u.clone())
            cw = torch.cos(w*self.dt).double()
            sw = torch.sin(w*self.dt).double()
            x = x + v*c*self.dt
            y = y + v*s*self.dt
            c = c*cw - s*sw
            s = s*cw + c*sw
            v = torch.clamp(v + u[0]*self.dt, -0.1, 0.3)
            w = torch.clamp(w + u[1]*self.dt, -np.pi/4, np.pi/4)
            pstate = state.clone()
            assert id(pstate) != id(state)
            state = torch.stack([x, y, c, s, v, w])
            states.append(state.clone())
        return u_seq, states
