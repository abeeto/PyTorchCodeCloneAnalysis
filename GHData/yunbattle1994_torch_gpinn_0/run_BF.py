import h5py
import numpy as np
import torch
import torch.nn as nn
from process_data import data_norm, data_sampler
from basic_model import gradients, DeepModel_single, DeepModel_multi
import visual_data
import matplotlib.pyplot as plt
import time
import os
import argparse
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_args():
    parser = argparse.ArgumentParser('GPINNs for Brinkman-Forchheimer model 3.3.1', add_help=False)
    parser.add_argument('--net_type', default='pinn', type=str, help="pinn or gpinn")
    parser.add_argument('--Layer_depth', default=3, type=int, help="Number of Layers depth")
    parser.add_argument('--Layer_width', default=20, type=int, help="Number of Layers width")
    parser.add_argument('--out_norm', default=False, type=bool, help="output fields normalization")
    parser.add_argument('--activation', default=nn.Tanh(), help="activation function")
    parser.add_argument('--epochs_adam', default=50000, type=int)
    parser.add_argument('--save_freq', default=5000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=1000, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=0, type=int, help="gpu id")
    parser.add_argument('--work_name', default='', type=str, help="work path to save files")
    parser.add_argument('--loss_weight', default=[1.0, 0.1, 1.0], type=list, help="loss weight for [eqs, geqs, bcs]")
    parser.add_argument('--Nx_EQs', default=10, type=int, help="xy sampling in for equation loss")
    parser.add_argument('--Nx_DTs', default=5, type=int, help="xy sampling in for data loss")
    return parser.parse_args()


class Net_single(DeepModel_single):
    def __init__(self, planes, data_norm, active):
        super(Net_single, self).__init__(planes, data_norm, active)
        self.v_e_ = nn.Parameter(torch.tensor([0.1, ]))

    def out_transform(self, inn_var, out_var):
        return torch.tanh(inn_var)*torch.tanh(1-inn_var) * out_var

    def equation(self, inn_var, out_var):
        dudx = gradients(out_var, inn_var)[:, (0,)]
        d2udx2 = gradients(dudx.sum(), inn_var)[:, (0,)]
        v_e = torch.log(torch.exp(self.v_e_) + 1) * 0.1
        eqs = -v_e/e * d2udx2 + v * out_var/K - g
        g_eqs = torch.tensor(0.0, dtype=torch.float32, device=device)

        if opts.net_type == 'gpinn':
            d3udx3 = gradients(d2udx2.sum(), inn_var)[:, (0,)]
            g_eqs = -v_e/e * d3udx3 + v/K * dudx

        return eqs, g_eqs, dudx

    def get_params(self):
        return torch.log(torch.exp(self.v_e_) + 1) * 0.1



def train(inn_var, BCs, ICs, out_true, model, Loss, optimizer, scheduler, log_loss, opts):

    inn_EQs = torch.linspace(BCs[0], BCs[1], opts.Nx_EQs+2, dtype=torch.float32)[1:-1, None].to(device)
    inn_DTs = inn_var.to(device) # 监督测点
    out_DTs = out_true.to(device)
    inn_BCs = torch.tensor(BCs, dtype=torch.float32).to(device)[:, None]  # 监督测点

    def closure():

        optimizer.zero_grad()

        inn_EQs.requires_grad_(True)
        out_EQs__ = model(inn_EQs, is_norm=opts.out_norm)
        out_EQs_ = model.out_transform(inn_EQs, out_EQs__)
        res_EQs, res_GEQs, res_DUXs = model.equation(inn_EQs, out_EQs_)
        
        inn_DTs.requires_grad_(False)
        out_DTs__ = model(inn_DTs, is_norm=opts.out_norm)
        out_DTs_ = model.out_transform(inn_DTs, out_DTs__)
        
        inn_BCs.requires_grad_(False)
        out_BCs__ = model(inn_BCs, is_norm=opts.out_norm)
        out_BCs_ = model.out_transform(inn_BCs, out_BCs__)

        data_loss = Loss(out_DTs_, out_DTs)  # 监督测点
        bcs_loss_1 = (out_BCs_**2).mean() #左右边界
        eqs_loss = (res_EQs**2).mean()   #方程损失
        geqs_loss = (res_GEQs ** 2).mean()  # 方程损失

        if opts.net_type == 'pinn':
            loss_batch = (data_loss) + eqs_loss
        else:
            loss_batch = (data_loss) + eqs_loss + geqs_loss * 0.1
        loss_batch.backward()
        para_loss = Loss(model.get_params(), torch.tensor(0.001, dtype=torch.float32).cuda())   #测试inverse
        log_loss.append([para_loss.item(), eqs_loss.item(), geqs_loss.item(), bcs_loss_1.item(), data_loss.item(),])
        return loss_batch

    optimizer.step(closure)
    scheduler.step()

def inference(inn_var, model):

    with torch.no_grad():
        out_pred = model(inn_var, is_norm=opts.out_norm)
        out_pred = model.out_transform(inn_var, out_pred)
    return out_pred


# 解析解 ve=1e-3
def sol(x):
    r = (v * e / (1e-3 * K)) ** (0.5)
    return g * K / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, dtype=np.float32, endpoint=False)
    yvals = sol(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))



if __name__ == '__main__':

    opts = get_args()
    g = 1
    v = 1e-3
    K = 1e-3
    e = 0.4
    H = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.device)  # 指定第一块gpu

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    Ns = [5, 10, 15, 20, 25, 30, 35]
    # Ns = [10,]
    for N in Ns:
        opts.Nx_EQs = N
        work_name = 'brinkman_forchheimer_1-' + opts.net_type + '-N_' + str(opts.Nx_EQs) + opts.work_name
        work_path = os.path.join('work', work_name)
        tran_path = os.path.join('work', work_name, 'train')
        isCreated = os.path.exists(tran_path)
        if not isCreated:
            os.makedirs(tran_path)

        # 将控制台的结果输出到a.log文件，可以改成a.txt
        sys.stdout = visual_data.Logger(os.path.join(work_path, 'train.log'), sys.stdout)

        print(opts)

        ob_x, ob_u = gen_traindata(opts.Nx_DTs)  # 生成监督测点
        input = torch.tensor(ob_x, dtype=torch.float32, device=device)
        output = torch.tensor(ob_u, dtype=torch.float32, device=device)
        input_norm = data_norm(ob_x, method='min-max')
        output_norm = data_norm(ob_u, method='min-max')
        BCs=[0, 1]

        L1Loss = nn.L1Loss()
        HBLoss = nn.SmoothL1Loss()
        L2Loss = nn.MSELoss()

        planes = [1,] + [opts.Layer_width] * opts.Layer_depth + [1,]

        Net_model = Net_single(planes=planes, data_norm=(input_norm, output_norm), active=opts.activation).to(device)
        # param_net = [p for p in Net_model.layers.parameters()]
        # param_ve = [Net_model.v_e_, ]
        Optimizer1 = torch.optim.Adam(Net_model.parameters(), lr=0.001, betas=(0.9, 0.999))
        # Optimizer1 = torch.optim.Adam([{'params': param_net, 'lr': 0.001, 'beta': {0.7, 0.9}},
        #                                {'params': param_ve, 'lr': 0.0003, 'beta': {0.7, 0.9}}])
        Optimizer2 = torch.optim.LBFGS(Net_model.parameters(), lr=1, max_iter=100, history_size=50,)
        Boundary_epoch1 = [opts.epochs_adam*6/10, opts.epochs_adam*9/10]
        Boundary_epoch2 = [opts.epochs_adam*11/10, opts.epochs_adam*12/10]

        Scheduler1 = torch.optim.lr_scheduler.MultiStepLR(Optimizer1, milestones=Boundary_epoch1, gamma=0.1)
        Scheduler2 = torch.optim.lr_scheduler.MultiStepLR(Optimizer2, milestones=Boundary_epoch2, gamma=0.1)
        Visual = visual_data.matplotlib_vision('/', field_name=('u',), input_name=('x', ))
        Visual.font['size'] = 20

        star_time = time.time()
        start_epoch=0
        log_loss=[]
        log_par = []
        log_L2 = []
        """load a pre-trained model"""
        start_epoch, log_loss = Net_model.loadmodel(os.path.join(work_path, 'latest_model.pth'))

        for i in range(start_epoch):
            #  update the learning rate for start_epoch times
            Scheduler1.step()

            # Training
        for iter in range(start_epoch, opts.epochs_adam):

            if iter < opts.epochs_adam:
                train(input, BCs, None, output, Net_model, L2Loss, Optimizer1, Scheduler1, log_loss, opts)
                learning_rate = Optimizer1.state_dict()['param_groups'][0]['lr']

                log_par.append(Net_model.get_params().detach().cpu().numpy())
            else:
                train(input, BCs, None, output,Net_model, L2Loss, Optimizer1, Scheduler2, log_loss, opts)
                learning_rate = Optimizer2.state_dict()['param_groups'][0]['lr']

            input_test = np.linspace(BCs[0], BCs[1], 1002, dtype=np.float32, )[:, None]
            output_test = sol(input_test)
            output_pred_ = inference(torch.tensor(input_test, dtype=torch.float32).to(device), Net_model)
            output_pred = output_pred_.detach().cpu().numpy()
            L2_u = np.mean(np.linalg.norm(output_pred - output_test) / np.linalg.norm(output_test))
            L2_ve = np.mean(np.linalg.norm(log_par[-1] - 0.001) / np.linalg.norm(0.001))
            log_L2.append([L2_ve, L2_u])

            if iter > 0 and iter % opts.print_freq == 0:

                print('iter: {:6d}, lr: {:.1e}, cost: {:.2f}, pred_ve: {:.2e} , true_ve: 1e-3\n'
                      'para_loss: {:.2e}, EQs_loss: {:.2e}, GEQs_loss: {:.2e}, BCs_loss: {:.2e}, DTs_loss_0: {:.2e}'.
                      format(iter, learning_rate, time.time() - star_time, log_par[-1][0],
                             log_loss[-1][0], log_loss[-1][1], log_loss[-1][2],
                             log_loss[-1][3], log_loss[-1][4]))

                plt.figure(100, figsize=(20, 15))
                plt.clf()
                plt.subplot(2, 1, 1)
                Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'para_loss')
                Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'eqs_loss')
                Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'geqs_loss')
                plt.subplot(2, 1, 2)
                Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'BCS_loss')
                Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'DTS_loss')
                plt.savefig(os.path.join(tran_path, 'log_loss.svg'))


                plt.figure(5, figsize=(20, 15))
                plt.clf()
                Visual.plot_loss(np.arange(len(log_par)), np.array(log_par)[:, 0], 'GPINN')
                Visual.plot_loss(np.arange(len(log_par)), np.ones(len(log_par))*0.001, 'EXACT')
                plt.xlabel("Epoch")
                plt.ylabel("v_e")
                plt.savefig(os.path.join(tran_path, 'log_ve.svg'))


                plt.figure(4, figsize=(20, 15))
                plt.clf()
                Visual.plot_value(input_test, output_test, 'EXACT')
                Visual.plot_value(input_test, output_pred, 'GPINN')
                plt.plot(ob_x, ob_u, 'ko')
                plt.xlabel("x")
                plt.ylabel("u")
                plt.savefig(os.path.join(tran_path, 'true_pred_u.svg'))

                plt.figure(2, figsize=(20, 15))
                plt.clf()
                Visual.plot_loss(np.arange(len(log_L2)), np.array(log_L2)[:, 1], 'GPINN')
                plt.ylabel("L2_u")
                plt.savefig(os.path.join(tran_path, 'L2_u.svg'))

                plt.figure(1, figsize=(20, 15))
                plt.clf()
                Visual.plot_loss(np.arange(len(log_L2)), np.array(log_L2)[:, 0], 'GPINN')
                plt.ylabel("L2_ve")
                plt.savefig(os.path.join(tran_path, 'L2_ve.svg'))

                star_time = time.time()


            if iter > 0 and iter % opts.save_freq == 0:

                torch.save({'epoch': iter, 'model': Net_model.state_dict(), 'log_loss': log_loss, 'par_loss': log_par},
                           os.path.join(work_path, 'latest_model.pth'), )
