import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.io  # loads .mat files
import os
from pyDOE import lhs
import shutil

import sobol  # https://github.com/DavidWalz/sobol

import random
from datetime import datetime
from enum import Enum


# GOAL: 2 training: with and without batch
# Enums ################################################################################################################

# Activation Enum
class Activation(Enum):
    Tanh = 1
    Relu = 2
    Leaky_Relu = 3
    Sigmoid = 4
    Soft_Plus = 5


# Initial_Weight Enum
class Initial_Weight(Enum):
    Xavier_Normal = 1
    Xavier_Uniform = 2
    Kaiming_Normal = 3
    Kaiming_Uniform = 4


class Sample_Method(Enum):
    Uniform = 1
    Lhs = 2
    Sobol = 3


# Variables ############################################################################################################

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # specifies run device for more optimum runtime

input_size = 2
output_size = 2
hidden_size = 25
residual_block_size = 6

n_x = 100
n_y = 100
n = 1000
n_bc = 100

lam = 1.0
epochs_Adam = 2000
epochs_LBFGS = 200
criterion = nn.MSELoss()

learning_rate = 1e-3
step_size = 200
gamma = 0.999

max_iter = 10
tolerance_grad = 1.e-12
tolerance_change = 1.e-15

seed = 1234

print_epoch_Adam_interval = 50
print_epoch_LBFGS_interval = 1

print_dataset = True
print_testset = True
plot_loss_and_corrects = True
train_anyway = True

save_path_name = "saved_models_PINN"
save_model_name = "PINN_model_"
exact_mat_filename = 'Umut_Hoca/NLS.mat'

log_filename = "log_PINN.txt"


# Global Functions #####################################################################################################

def get_log_str(sample_method):
    log_str = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\ndevice:" + str(device) + ", input_size:" + str(input_size) + ", output_size:" + str(output_size) + ", hidden_size:" + str(
        hidden_size) + ", residual_block_size:" + str(residual_block_size) \
              + ", sample_method:" + str(sample_method) + ", n_x:" + str(n_x) + ", n_y:" + str(n_y) + ", n:" + str(n) + ", n_bc:" + str(n_bc) + ", lam:" + str(lam) + ", epochs_Adam:" + str(
        epochs_Adam) \
              + ", epochs_LBFGS:" + str(epochs_LBFGS) + ", learning_rate:" + str(learning_rate) + ", step_size:" + str(step_size) + ", gamma:" + str(gamma) + ", seed:" + str(seed)

    return log_str


def get_file_str(sample_method):
    filename_str = str(device) + str(input_size) + str(output_size) + str(hidden_size) + str(residual_block_size) + str(sample_method.value) + str(n_x) + str(n_y) + str(n) + str(n_bc) + str(
        lam) + str(epochs_Adam) + str(epochs_LBFGS) + str(learning_rate) + str(step_size) + str(gamma)
    return filename_str

def activation(name):
    if name == Activation.Tanh:
        return nn.Tanh()
    elif name == Activation.Relu:
        return nn.ReLU(inplace=True)
    elif name == Activation.Leaky_Relu:
        return nn.LeakyReLU(inplace=True)
    elif name == Activation.Sigmoid:
        return nn.Sigmoid()
    elif name == Activation.Soft_Plus:
        return nn.Softplus()
    else:
        raise ValueError(f'unknown activation function: {name}')


# derivative of y respect to x
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)


def get_init_weight(name):
    if name == Initial_Weight.Xavier_Normal:
        return nn.init.xavier_normal_
    elif name == Initial_Weight.Xavier_Uniform:
        return nn.init.xavier_uniform_
    elif name == Initial_Weight.Kaiming_Normal:
        return nn.init.kaiming_normal_
    elif name == Initial_Weight.Kaiming_Uniform:
        return nn.init.kaiming_uniform_
    else:
        raise ValueError(f'unknown initialization weight: {name}')


def numpy_to_tensor(x):
    lst = []
    for item in x:
        lst.append(torch.from_numpy(np.asarray(item)).float())
    return lst


def assign_to_device(x):
    lst = []
    for item in x:
        if not isinstance(item, list):
            lst.append(item.to(device))
        else:
            lst.append(assign_to_device(item))
    return tuple(lst)


def enter_log(text, log_str, filename_str, header=False):
    file = open(log_filename, "a")
    if header:
        file.write("\n" + log_str + "\n" + "Filename: " + filename_str + "\n")
    file.write(text + "\n")
    file.close()


def save_model(model, filename_str):
    # Path to save the model
    if not os.path.exists(save_path_name):
        os.mkdir(save_path_name)

    path = os.path.join(save_path_name, filename_str)
    if not os.path.isfile(path):
        torch.save(model, path)


def get_model(filename_str):
    if not os.path.exists(save_path_name):
        return None

    path = os.path.join(save_path_name, filename_str)
    if not os.path.isfile(path):
        return None

    model = torch.load(path)
    return model


def plot_loss_epoch(epochs, losses, validation_losses, method):

    plt.title(f'Epoch vs Loss ({method})')
    plt.plot(range(epochs), losses, label="training loss of " + str(method))
    plt.plot(np.array(validation_losses)[:, 0], np.array(validation_losses)[:, 1], label="validation loss of " + str(method))
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(color='gray', linestyle='-', linewidth=0.2)
    # plt.show()


########################################################################################################################

class Residual_Block(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation_name=Activation.Tanh):
        super().__init__()
        self.activation1 = activation(activation_name)
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.activation2 = activation(activation_name)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x):
        i = x
        x = self.linear1(self.activation1(x))
        x = self.linear2(self.activation2(x))
        return i + x


########################################################################################################################

class Residual_DNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, residual_blocks, activation_name=Activation.Tanh, init_name=Initial_Weight.Kaiming_Normal):

        super().__init__()

        model = nn.Sequential()
        model.add_module('fc_first', nn.Linear(input_size, hidden_size, bias=True))

        for i in range(residual_blocks):
            residual_block = Residual_Block(hidden_size, hidden_size, hidden_size, activation_name=activation_name)
            model.add_module(f'res_block{i + 1}', residual_block)

        model.add_module('act_last', activation(activation_name))
        model.add_module('fc_last', nn.Linear(hidden_size, output_size, bias=True))

        self.model = model
        self.init_weight(init_name)

    def forward(self, x):
        return self.model(x)

    def init_weight(self, name):
        nn_init = get_init_weight(name)

        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)

    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params


########################################################################################################################

# Physics Informed Neural Network Implementation using Residual DNN
class Residual_PINN_Schrodinger(Residual_DNN):

    def __init__(self, input_size, hidden_size, output_size, residual_blocks, activation_name=Activation.Tanh, init_name=Initial_Weight.Xavier_Normal):
        super().__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size, residual_blocks=residual_blocks, activation_name=activation_name, init_name=init_name)

    def forward(self, x, x_bc_1=None, x_bc_2=None, x_init=None):
        # h(x, t) = [u(x, t), v(x, t)]      where h is the wave function, u is its real part, and v is its imaginary part
        x.requires_grad_(True)
        H = super().forward(x)

        U = H[:, [0]]
        V = H[:, [1]]

        grad_u = grad(U, x)[0]
        grad_v = grad(V, x)[0]

        U_x = grad_u[:, [1]]
        U_t = grad_u[:, [0]]
        V_x = grad_v[:, [1]]
        V_t = grad_v[:, [0]]

        U_xx = grad(U_x, x)[0][:, [1]]
        V_xx = grad(V_x, x)[0][:, [1]]
        x.detach_()

        # Schrodinger function (real and imaginary parts are separated)
        C = 0.5
        F_U = V_t - C * U_xx - U * (U ** 2 + V ** 2)  # f_u = V_t - 0.5 U_xx - u (u² + v²)
        F_V = U_t + C * V_xx + V * (U ** 2 + V ** 2)  # f_v = U_t + 0.5 V_xx - v (u² + v²)

        if x_init is not None:
            g_init = super().forward(x_init)
            x_bc_1.requires_grad_(True)
            x_bc_2.requires_grad_(True)
            h_1 = super().forward(x_bc_1)
            h_2 = super().forward(x_bc_2)

            ux_1 = grad(h_1[:, [0]], x_bc_1)[0][:, [1]]
            ux_2 = grad(h_2[:, [0]], x_bc_2)[0][:, [1]]
            vx_1 = grad(h_1[:, [1]], x_bc_1)[0][:, [1]]
            vx_2 = grad(h_2[:, [1]], x_bc_2)[0][:, [1]]
            x_bc_1.detach_()
            x_bc_2.detach_()

            g_init_u = g_init[:, [0]]
            g_init_v = g_init[:, [1]]
            g_bc_1_u = h_1[:, [0]] - h_2[:, [0]]
            g_bc_1_v = h_1[:, [1]] - h_2[:, [1]]
            g_bc_2_u = ux_1 - ux_2
            g_bc_2_v = vx_1 - vx_2

            return H, (F_U, F_V), (g_bc_1_u, g_bc_1_v), (g_bc_2_u, g_bc_2_v), (g_init_u, g_init_v)

        return H, F_U, F_V


########################################################################################################################


# Description of Schrodinger problem
# includes, f function (u and v) and initial conditions (t=0, x=-A, x=A)
class Problem_Schrodinger:

    def __init__(self, domain):
        self.domain = domain

    def f(self, x):
        number, _ = x.shape
        U = np.zeros((number, 1))
        V = np.zeros((number, 1))
        return U, V

    def initial(self, x_init):
        number, _ = x_init.shape
        U = 2.0 / np.cosh(x_init[:, [1]])
        V = np.zeros((number, 1))
        return U, V

    def lower_boundary(self, x_lb):
        number, _ = x_lb.shape
        U = np.zeros((number, 1))
        V = np.zeros((number, 1))
        return U, V

    def upper_boundary(self, x_up):
        number, _ = x_up.shape
        U = np.zeros((number, 1))
        V = np.zeros((number, 1))
        return U, V


########################################################################################################################

# creates trainset for Schrodinger
class Trainset_Schrodinger:

    def __init__(self, problem, method=Sample_Method.Sobol, N_x=None, N_y=None, N=None, N_boundary=None, purpose=None):
        self.problem = problem
        self.method = method
        self.args = (N_x, N_y, N, N_boundary)
        self.purpose = purpose

    def get_trainset(self):

        if self.method == Sample_Method.Uniform:
            n_x, n_y = self.args[0], self.args[1]
            x, x_lb, x_ub, x_init = self.uniform_sample(n_x, n_y, self.problem.domain)

        elif self.method == Sample_Method.Lhs:
            # print('self', self)
            n, n_bc = self.args[2], self.args[3]
            x, x_lb, x_ub, x_init = self.lhs_sample(n, n_bc, self.problem.domain)
            # print("b:", x, x_lb, x_ub, x_init)

        elif self.method == Sample_Method.Sobol:
            n, n_bc = self.args[2], self.args[3]
            x, x_lb, x_ub, x_init = self.sobol_sample(n, n_bc, self.problem.domain)

        else:
            raise ValueError(f'unknown method: {self.method}')

        f = numpy_to_tensor(self.problem.f(x))
        init = numpy_to_tensor(self.problem.initial(x_init))
        lower_boundary = numpy_to_tensor(self.problem.lower_boundary(x_lb))
        upper_boundary = numpy_to_tensor(self.problem.lower_boundary(x_ub))

        if print_dataset:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x[:, 0], x[:, 1], facecolor='r', s=10)  # sampled data
            ax.scatter(x_lb[:, 0], x_lb[:, 1], facecolor='b', s=10)  # lower bound
            ax.scatter(x_ub[:, 0], x_ub[:, 1], facecolor='b', s=10)  # upper bound
            ax.scatter(x_init[:, 0], x_init[:, 1], facecolor='b', s=10)  # initial (t = 0)
            ax.set_xlim(-0.05, 0.05 + np.pi / 2)
            ax.set_ylim(-5.5, 5.5)
            ax.set_xlabel('$t$')
            ax.set_ylabel('$x$')

            name = Sample_Method(self.method).name
            ax.set_title(f'{self.purpose} ({name})', fontsize=10)
            plt.grid()

            # plt.savefig(f'{self.purpose} ({name}).jpg', dpi=500, bbox_inches='tight')
            plt.show()

        return torch.from_numpy(np.asarray(x)).float(), torch.from_numpy(np.asarray(x_lb)).float(), torch.from_numpy(np.asarray(x_ub)).float(), torch.from_numpy(
            np.asarray(x_init)).float(), f, init, lower_boundary, upper_boundary

    def uniform_sample(self, n_x, n_y, domain):
        t_min, t_max, x_min, x_max = domain
        t = np.linspace(t_min, t_max, n_x)
        x = np.linspace(x_min, x_max, n_y)
        t, x = np.meshgrid(t, x)
        xy = np.hstack((t.reshape(t.size, -1), x.reshape(x.size, -1)))

        mask0 = (x_max - xy[:, 1]) == 0
        mask1 = (xy[:, 1] - x_min) == 0
        mask2 = (xy[:, 0] - t_min) == 0
        mask3 = mask0 + mask1 + mask2
        x_bc_1 = xy[mask0]
        x_bc_2 = xy[mask1]
        x_init = xy[mask2]
        x = xy[np.logical_not(mask3)]

        return x, x_bc_1, x_bc_2, x_init

    def lhs_sample(self, n, n_bc, domain):
        t_min, t_max, x_min, x_max = domain

        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_max])
        x = lb + (ub - lb) * lhs(2, n)

        sample = lhs(2, n_bc // 2)
        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_min])
        x_bc_1 = lb + (ub - lb) * sample

        lb = np.array([t_min, x_max])
        ub = np.array([t_max, x_max])
        x_bc_2 = lb + (ub - lb) * sample

        lb = np.array([t_min, x_min])
        ub = np.array([t_min, x_max])
        x_init = lb + (ub - lb) * lhs(2, n_bc // 2)
        return x, x_bc_1, x_bc_2, x_init

    def sobol_sample(self, n, n_bc, domain):
        t_min, t_max, x_min, x_max = domain

        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_max])
        x = lb + (ub - lb) * sobol.sample(dimension=2, n_points=n)

        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_min])
        x_bc_1 = lb + (ub - lb) * sobol.sample(dimension=2, n_points=n_bc)

        lb = np.array([t_min, x_max])
        ub = np.array([t_max, x_max])
        x_bc_2 = lb + (ub - lb) * sobol.sample(dimension=2, n_points=n_bc)

        lb = np.array([t_min, x_min])
        ub = np.array([t_min, x_max])
        x_init = lb + (ub - lb) * sobol.sample(dimension=2, n_points=n_bc)
        return x, x_bc_1, x_bc_2, x_init


########################################################################################################################

# creates testset for Schrodinger
class Testset_Schrodinger:

    def __init__(self, problem, method=Sample_Method.Sobol, N_x=None, N_y=None, N=None, N_boundary=None):
        self.problem = problem
        self.method = method
        self.args = (N_x, N_y, N, N_boundary)

    def get_testset(self):
        data = scipy.io.loadmat(exact_mat_filename)
        t = data['tt'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        Exact = data['uu']
        Exact_u = np.real(Exact)
        Exact_v = np.imag(Exact)
        Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)
        t, x = np.meshgrid(t, x)
        X = np.hstack((t.reshape(t.size, -1), x.reshape(x.size, -1)))

        if print_testset:
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.scatter(X[:, 0], X[:, 1], facecolor='r', s=0.01)
            ax.set_xlim(-0.01, np.pi / 2 + 0.01)
            ax.set_ylim(-5.1, 5.1)
            ax = fig.add_subplot(122, projection='3d')
            ax.plot_surface(t, x, Exact_h, cmap=cm.coolwarm)
            fig.suptitle('Test Set from mat File', fontsize=10)

            # plt.savefig('Test Set from mat File.jpg', dpi=500, bbox_inches='tight')
            plt.show()

        X = torch.from_numpy(X).float()
        return X, t, x, Exact_h


########################################################################################################################

# trains Residual_PINN_Schrodinger using Trainset
class Trainer_Schrodinger:

    def __init__(self, problem, model, trainset, validationset, sample_method):

        self.problem = problem
        self.model = model
        self.trainset = trainset
        self.validationset = validationset
        self.criterion = criterion
        self.sample_method = sample_method

        # optimizers (Adam and LBFGS) which will minimize loss function
        self.optimizer_Adam = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer_LBFGS = optim.LBFGS(self.model.parameters(), max_iter=max_iter, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change)

        # active learning rate scheduler which will decrease the learning rate as the epoch num increases
        self.lr_scheduler = StepLR(self.optimizer_Adam, step_size=step_size, gamma=gamma)

        self.model.to(device)
        self.model.zero_grad()

        # get trainset
        x, x_lb, x_ub, x_init, f, init, lower_boundary, upper_boundary = trainset.get_trainset()
        self.x, self.x_lb, self.x_ub, self.x_init, self.f, self.init, self.lower_boundary, self.upper_boundary = assign_to_device([x, x_lb, x_ub, x_init, f, init, lower_boundary, upper_boundary])

        # get validationset
        x_val, x_lb_val, x_ub_val, x_init_val, f_val, init_val, lower_boundary_val, upper_boundary_val = validationset.get_trainset()
        self.x_val, self.x_lb_val, self.x_ub_val, self.x_init_val, self.f_val, self.init_val, self.lower_boundary_val, self.upper_boundary_val = assign_to_device(
            [x_val, x_lb_val, x_ub_val, x_init_val, f_val, init_val, lower_boundary_val, upper_boundary_val])

        # print('trainset:\n\n', self.x, self.x_lb, self.x_ub, self.x_init, self.f, self.init, self.lower_boundary, self.upper_boundary)

        # print('validationset:\n\n', self.x_val, self.x_lb_val, self.x_ub_val, self.x_init_val, self.f_val, self.init_val, self.lower_boundary_val, self.upper_boundary_val)

    def train(self):

        losses = []
        validation_losses = []

        for epoch in range(epochs_Adam):

            # train Adam epochs
            loss, loss_f, loss2 = self.train_Adam()
            losses.append(loss)

            # print epoch info
            if (epoch + 1) % print_epoch_Adam_interval == 0:
                print('Adam Epoch ', epoch + 1, '/', epochs_Adam + epochs_LBFGS, f', Loss: {loss:4e}', ', lr:', self.lr_scheduler.get_last_lr()[0])
                enter_log('Adam Epoch  ' + str(epoch + 1) + '/' + str(epochs_Adam + epochs_LBFGS) + f', Loss: {loss:4e}' + ', lr:' + str(self.lr_scheduler.get_last_lr()[0]), get_log_str(self.sample_method), get_file_str(self.sample_method))

                # validation
                loss_validation = self.validation()

                validation_losses.append((epoch, loss_validation))
                print('Valid Epoch', epoch + 1, '/', epochs_Adam + epochs_LBFGS, f', Loss: {loss_validation:4e}')
                enter_log('Valid Epoch ' + str(epoch + 1) + '/' + str(epochs_Adam + epochs_LBFGS) + f', Loss: {loss_validation:4e}', get_log_str(self.sample_method), get_file_str(self.sample_method))


        for epoch in range(epochs_LBFGS):

            # train LBFGS epochs
            loss, loss_f, loss2 = self.train_LBFGS()
            losses.append(loss)

            # print epoch info
            if (epoch + 1) % print_epoch_LBFGS_interval == 0:
                print('LBFGS Epoch', epoch + 1 + epochs_Adam, '/', epochs_Adam + epochs_LBFGS, f', Loss: {loss:4e}')
                enter_log('LBFGS Epoch ' + str(epoch + 1 + epochs_Adam) + '/' + str(epochs_Adam + epochs_LBFGS) + f', Loss: {loss:4e}', get_log_str(self.sample_method), get_file_str(self.sample_method))

                # validation
                loss_validation = self.validation()
                validation_losses.append((epoch+epochs_Adam, loss_validation))
                print('Valid Epoch', epoch + 1 + epochs_Adam, '/', epochs_Adam + epochs_LBFGS, f', Loss: {loss_validation:4e}')
                enter_log('Valid Epoch ' + str(epoch + 1 + epochs_Adam) + '/' + str(epochs_Adam + epochs_LBFGS) + f', Loss: {loss_validation:4e}', get_log_str(self.sample_method), get_file_str(self.sample_method))

        return losses, validation_losses


    def train_Adam(self):

        # optimizer.zero_grad()
        self.optimizer_Adam.zero_grad()

        # make prediction
        _, f_pred, g_bc_1, g_bc_2, g_init = self.model.forward(self.x, x_bc_1=self.x_lb, x_bc_2=self.x_ub, x_init=self.x_init)

        loss1_u = self.criterion(f_pred[0], self.f[0])
        loss1_v = self.criterion(f_pred[1], self.f[1])

        loss1 = loss1_u + loss1_v

        loss2_bc_1_u = self.criterion(g_bc_1[0], self.lower_boundary[0])
        loss2_bc_1_v = self.criterion(g_bc_1[1], self.lower_boundary[1])

        loss2_bc_2_u = self.criterion(g_bc_2[0], self.upper_boundary[0])
        loss2_bc_2_v = self.criterion(g_bc_2[1], self.upper_boundary[1])

        loss2_init_u = self.criterion(g_init[0], self.init[0])
        loss2_init_v = self.criterion(g_init[1], self.init[1])

        loss2 = loss2_bc_1_u + loss2_bc_1_v + loss2_bc_2_u + loss2_bc_2_v + loss2_init_u + loss2_init_v

        loss = loss1 + lam * loss2

        loss.backward()
        self.optimizer_Adam.step()
        self.lr_scheduler.step()

        return loss.item(), loss1.item(), loss2.item()

    def train_LBFGS(self):

        def closure():
            if torch.is_grad_enabled():
                self.optimizer_LBFGS.zero_grad()
            _, f_pred, g_bc_1, g_bc_2, g_init = self.model.forward(self.x, x_bc_1=self.x_lb, x_bc_2=self.x_ub, x_init=self.x_init)

            loss1_u = self.criterion(f_pred[0], self.f[0])
            loss1_v = self.criterion(f_pred[1], self.f[1])

            loss1 = loss1_u + loss1_v

            loss2_bc_1_u = self.criterion(g_bc_1[0], self.lower_boundary[0])
            loss2_bc_1_v = self.criterion(g_bc_1[1], self.lower_boundary[1])

            loss2_bc_2_u = self.criterion(g_bc_2[0], self.upper_boundary[0])
            loss2_bc_2_v = self.criterion(g_bc_2[1], self.upper_boundary[1])

            loss2_init_u = self.criterion(g_init[0], self.init[0])
            loss2_init_v = self.criterion(g_init[1], self.init[1])

            loss2 = loss2_bc_1_u + loss2_bc_1_v + loss2_bc_2_u + loss2_bc_2_v + loss2_init_u + loss2_init_v

            loss = loss1 + lam * loss2

            if loss.requires_grad:
                loss.backward()
            return loss

        self.optimizer_LBFGS.step(closure)
        # loss = closure()

        # only used to compute loss_int and loss_bc1 for monitoring
        _, f_pred, g_bc_1, g_bc_2, g_init = self.model.forward(self.x, x_bc_1=self.x_lb, x_bc_2=self.x_ub, x_init=self.x_init)

        loss1_u = self.criterion(f_pred[0], self.f[0])
        loss1_v = self.criterion(f_pred[1], self.f[1])

        loss1 = loss1_u + loss1_v

        loss2_bc_1_u = self.criterion(g_bc_1[0], self.lower_boundary[0])
        loss2_bc_1_v = self.criterion(g_bc_1[1], self.lower_boundary[1])

        loss2_bc_2_u = self.criterion(g_bc_2[0], self.upper_boundary[0])
        loss2_bc_2_v = self.criterion(g_bc_2[1], self.upper_boundary[1])

        loss2_init_u = self.criterion(g_init[0], self.init[0])
        loss2_init_v = self.criterion(g_init[1], self.init[1])

        loss2 = loss2_bc_1_u + loss2_bc_1_v + loss2_bc_2_u + loss2_bc_2_v + loss2_init_u + loss2_init_v

        loss = loss1 + lam * loss2

        return loss.item(), loss1.item(), loss2.item()

    def validation(self):

        # setup validation
        self.model.eval()

        # get prediction
        # each prediction consists of 2 elements u and v
        _, f_pred, g_bc_1, g_bc_2, g_init = self.model.forward(self.x_val, x_bc_1=self.x_lb_val, x_bc_2=self.x_ub_val, x_init=self.x_init_val)

        loss1_u = self.criterion(f_pred[0], self.f_val[0])
        loss1_v = self.criterion(f_pred[1], self.f_val[1])

        loss1 = loss1_u + loss1_v

        loss2_bc_1_u = self.criterion(g_bc_1[0], self.lower_boundary_val[0])
        loss2_bc_1_v = self.criterion(g_bc_1[1], self.lower_boundary_val[1])

        loss2_bc_2_u = self.criterion(g_bc_2[0], self.upper_boundary_val[0])
        loss2_bc_2_v = self.criterion(g_bc_2[1], self.upper_boundary_val[1])

        loss2_init_u = self.criterion(g_init[0], self.init_val[0])
        loss2_init_v = self.criterion(g_init[1], self.init_val[1])

        loss2 = loss2_bc_1_u + loss2_bc_1_v + loss2_bc_2_u + loss2_bc_2_v + loss2_init_u + loss2_init_v

        loss = loss1 + lam * loss2

        self.model.train()
        return loss.item()


########################################################################################################################

# tests Residual_PINN_Schrodinger using Testset
class Tester_Schrodinger:

    def __init__(self, problem, model, testset):
        self.problem = problem
        self.model = model
        self.testset = testset
        self.criterion = criterion
        self.X, self.t, self.x, self.h_tar = testset.get_testset()

        self.X = self.X.to(device)

    # tests model using mat file
    def test(self, sample_method):

        self.model.eval()

        # make prediction
        H_pred, _, _ = self.model.forward(self.X)
        H_pred = H_pred.detach().cpu().numpy()
        H_pred = np.sqrt(H_pred[:, [0]] ** 2 + H_pred[:, [1]] ** 2)
        H_pred = H_pred.reshape(self.x.shape)

        fig = plt.figure()
        ax = fig.add_subplot(131, projection='3d')
        ax.plot_surface(self.t, self.x, self.h_tar, cmap=cm.coolwarm)
        ax.set_title('Exact', fontsize=10)
        ax = fig.add_subplot(132, projection='3d')
        ax.plot_surface(self.t, self.x, H_pred, cmap=cm.coolwarm)
        ax.set_title('Predicted', fontsize=10)
        ax = fig.add_subplot(133, projection='3d')
        ax.plot_surface(self.t, self.x, np.abs(H_pred - self.h_tar), cmap=cm.coolwarm)
        ax.set_title('Difference (Error)', fontsize=10)
        ax.set_xlim(-0.01, 0.01 + np.pi / 2)
        ax.set_ylim(-5.1, 5.1)

        # plt.savefig(f'Exact, Predicted, and Difference{sample_method}.jpg', dpi=500, bbox_inches='tight')
        plt.show()


    def plot(self, sample_method):
        self.model.eval()
        uv_pred, _, _ = self.model.forward(self.X)
        uv_pred = uv_pred.detach().cpu().numpy()
        uv_pred = np.sqrt(uv_pred[:, [0]] ** 2 + uv_pred[:, [1]] ** 2)
        h = uv_pred.reshape(self.x.shape)

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.axis('off')

        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])

        H = ax.imshow(h, interpolation='nearest', cmap='YlGnBu',
                      extent=self.problem.domain,
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(H, cax=cax)

        fig.suptitle('Non-Linear Schrodinger Prediction vs Exact', fontsize=10)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('$|h(t,x)|$', fontsize=10)

        ####### Row 1: u(t,x) slices ##################
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

        ax = plt.subplot(gs1[0, 0])
        ax.plot(self.x[:, 0], self.h_tar[:, 75], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x[:, 0], h[:, 75], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.set_title('$t = 0.25$', fontsize=10)
        ax.axis('square')
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 5])

        ax = plt.subplot(gs1[0, 1])
        ax.plot(self.x[:, 0], self.h_tar[:, 80], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x[:, 0], h[:, 80], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.axis('square')
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 5])
        ax.set_title('$t = 0.50$', fontsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

        ax = plt.subplot(gs1[0, 2])
        ax.plot(self.x[:, 0], self.h_tar[:, 90], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x[:, 0], h[:, 90], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.axis('square')
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 5])
        ax.set_title('$t = 0.75$', fontsize=10)

        # plt.savefig(f'NonLinearSch{sample_method}.jpg', dpi=500, bbox_inches='tight')
        plt.show()


# MAIN #################################################################################################################

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create Schrodinger problem
    domain = (0, np.pi / 2, -5, 5)
    problem = Problem_Schrodinger(domain)

    # holds losses for each method
    losses_method = {}

    # determine sample method
    for sample_method in Sample_Method:
        print('\nTraining with sample method:', sample_method)
        enter_log("\ntraining with sample method:" + str(sample_method), get_log_str(sample_method), get_file_str(sample_method), True)

        model = get_model(get_file_str(sample_method))
        if model is None or train_anyway:
            # create model
            model = Residual_PINN_Schrodinger(input_size, hidden_size, output_size, residual_block_size, activation_name=Activation.Tanh, init_name=Initial_Weight.Xavier_Normal)
            # print(model)
            enter_log("Model initialized.", get_log_str(sample_method), get_file_str(sample_method))

            # trainset
            trainset = Trainset_Schrodinger(problem, method=sample_method, N_x=n_x, N_y=n_y, N=n, N_boundary=n_bc, purpose='Training Set')
            # print(trainset.get_trainset())
            enter_log("Trainset is created.", get_log_str(sample_method), get_file_str(sample_method))

            # has to be Lhs
            validationset = Trainset_Schrodinger(problem, method=Sample_Method.Lhs, N=100, N_boundary=100, purpose='Validation Set')
            # print("a:", validationset.get_trainset())
            enter_log("Validationset is created.", get_log_str(sample_method), get_file_str(sample_method))

            # trainer
            trainer = Trainer_Schrodinger(problem, model, trainset, validationset, sample_method)
            enter_log("Training begins...", get_log_str(sample_method), get_file_str(sample_method))

            # save losses
            losses, validation_losses = trainer.train()
            losses_method[sample_method] = (losses, validation_losses)

            # save model
            save_model(model, get_file_str(sample_method))
            enter_log("Model saved.", get_log_str(sample_method), get_file_str(sample_method))
        else:
            print("model is loaded from a saved file.")


        # testset
        testset = Testset_Schrodinger(problem, method=sample_method, N_x=n_x, N_y=n_y, N=n, N_boundary=n_bc)
        # X, t, x, Exact_h = testset.get_testset()
        # print('X:', X, '\nt:', t, '\nx:', x, '\nExact_h', Exact_h)

        # tester
        tester = Tester_Schrodinger(problem, model, testset)

        # plot tests
        tester.plot(sample_method)

        # test model
        tester.test(sample_method)

    # plot losses
    if len(losses_method) is not 0:
        for key, value in losses_method.items():
            (losses, validation_losses) = value
            plot_loss_epoch(epochs_Adam+epochs_LBFGS, losses, validation_losses, key)

        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        plt.xlim(0, 150)
        plt.ylim(0, 10)
        # plt.savefig('loss_comparison.jpg', dpi=500, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    main()

# LBFGS Epoch 2200 / 2200 Loss: 8.547882e-04
# Valid Epoch 2200 / 2200 Loss: 2.253222e-03
