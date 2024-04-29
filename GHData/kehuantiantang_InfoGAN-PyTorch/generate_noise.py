# coding=utf-8
import torch
import numpy as np

# def generate_noise(batch_size = 100, num_dis_c = params.num_dis_c, dis_c_dim = params.dis_c_dim, num_con_c = params.num_con_c, num_z = params.num_z):
#     # Fixed Noise
#     z = torch.randn(batch_size, num_z, 1, 1, device=device)
#     fixed_noise = z
#
#     idx = np.zeros((num_dis_c, batch_size))
#     if (num_dis_c != 0):
#         # idx = np.arange(dis_c_dim).repeat(batch_size / dis_c_dim)
#
#         dis_c = torch.zeros(batch_size, num_dis_c, dis_c_dim, device=device)
#
#         for i in range(num_dis_c):
#             idx[i] = np.arange(dis_c_dim).repeat(batch_size / dis_c_dim)
#
#             dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0
#
#         dis_c = dis_c.view(batch_size, -1, 1, 1)
#         print(dis_c)
#
#         fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)
#
#     if (num_con_c != 0):
#         con_c = torch.rand(batch_size, num_con_c, 1, 1, device=device) * 2 - 1
#         fixed_noise = torch.cat((fixed_noise, con_c), dim=1)
#
#     return fixed_noise
from torch.autograd import Variable

from config.config import get_params





def get_noise(params, device):
    # Fixed Noise
    z = torch.randn(100, params['num_z'], 1, 1, device=device)
    fixed_noise = z
    if (params['num_dis_c'] != 0):
        idx = np.arange(params['dis_c_dim']).repeat(10)
        dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
        for i in range(params['num_dis_c']):
            dis_c[torch.arange(0, 100), i, idx] = 1.0

        dis_c = dis_c.view(100, -1, 1, 1)

        fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

    if (params['num_con_c'] != 0):
        con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
        fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

    return {'fixed':fixed_noise}

# def get_noiseByContinuous_code(params, device, rows = 10, columns = 10):
#
#     batch_size = rows * columns
#
#     # noise variable
#     # z = torch.randn(100, params.num_z, 1, 1, device=device)
#     z = torch.zeros(100, params.num_z, 1, 1, device=device)
#     # z = torch.ones(batch_size, params.num_z, 1, 1, device=device)
#
#     # idx = np.arange(rows).repeat(columns)
#     idx = np.zeros((params.num_dis_c, batch_size))
#     dis_c = torch.zeros(batch_size, params.num_dis_c, params.dis_c_dim, device=device)
#
#     for i in range(params.num_dis_c):
#         idx[i] = np.arange(columns).repeat(rows)
#         dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0
#
#     # categorical code
#     cat = dis_c.view(batch_size, -1, 1, 1)
#
#     con_code = np.linspace(params.var.v1, params.var.v2, rows).reshape(1, -1)
#     con_code = np.repeat(con_code, columns, 0).reshape((-1, 1, 1))
#     # con = torch.from_numpy(con).float().to(device)
#     # con = con.view(-1, 1, 1, 1)
#     noises = {}
#     for i in range(params.num_con_c):
#         con = np.zeros((batch_size, params.num_con_c, 1, 1))
#         con[:, i, ...] = con_code
#         con =  torch.from_numpy(con.copy()).float().to(device)
#
#         noise =  torch.cat([z, cat, con], dim=1)
#         noises['v%d'%i] = noise
#
#     return noises

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)



def to_torch(array, device):
    return torch.from_numpy(array).float().to(device)

def get_noiseByContinuous_code(params, device, rows = 10, columns = 10):

    batch_size = rows * columns

    # noise variable
    z = np.ones((batch_size, params.num_z, 1, 1))


    dcs = {}
    for i in range(params.num_dis_c):
        dis_c = np.zeros((batch_size, params.num_dis_c, params.dis_c_dim))

        temp = np.zeros((batch_size, params.dis_c_dim))
        for j in range(params.dis_c_dim):

            temp[j*rows:j*rows+rows, j] = params.num_dis_c#float(i+1)

        dis_c[:, i, ...] = temp
        dcs['dcs%02d'%i] = np.reshape(dis_c, (batch_size, -1, 1, 1)).copy()

    # for _, d in dcs.items():
    #     print(np.reshape(d, (batch_size, -1)))

    ccs = {}
    con_code = np.linspace(params.var.v1, params.var.v2, rows).reshape(1, -1)
    con_code = np.repeat(con_code, columns, 0).reshape((-1, 1, 1))
    for i in range(params.num_con_c):
        con = np.zeros((batch_size, params.num_con_c, 1, 1))
        con[:, i, ...] = con_code

        ccs['ccs%02d'%i] = con.copy()

    # for _, c in ccs.items():
    #     print(np.reshape(c, (batch_size, -1)))

    noises = {}
    if len(ccs.keys()) > 0 and len(dcs.keys()) > 0:
        for dc_key, dc in dcs.items():
            for cc_key, cc in ccs.items():
               noises[dc_key + '_' + cc_key] = to_torch(np.concatenate((z, dc, cc), axis=1), device)

    elif len(ccs.keys()) > 0:
        for cc_key, cc in ccs.items():
            noises[cc_key] = to_torch(np.concatenate((z, cc), axis=1),device)
    elif len(dcs.keys()) > 0:
        for dc_key, dc in dcs.items():
            noises[dc_key] = to_torch(np.concatenate((z, dc), axis=1), device)

    return noises

if __name__ == '__main__':
    np.set_printoptions(threshold=1e6)
    params = get_params()
    for key, value in get_noiseByContinuous_code(params, 'cpu').items():
        print(key, value.size())
        # print(np.reshape(value, (100, -1)))
    # print(get_noiseByContinuous_code(params, 'cpu').keys())