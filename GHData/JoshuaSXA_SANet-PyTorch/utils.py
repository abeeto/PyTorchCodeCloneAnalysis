import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


def gaussian_filter(channel_size, kernel_size, sigma):
    gauss_1D = torch.Tensor([math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
    gauss_1D /= gauss_1D.sum()
    gauss_1D = gauss_1D.unsqueeze(1)
    gauss_2D = gauss_1D.mm(gauss_1D.t()).float().unsqueeze(0).unsqueeze(0)
    gauss_filter = Variable(gauss_2D.expand(channel_size, 1, kernel_size, kernel_size).contiguous())
    return gauss_filter


def ssim_index(pred, g_t, gauss_filter, kernel_size=11):
    mu_f = F.conv2d(pred, gauss_filter, padding=kernel_size//2, groups=1)
    mu_y = F.conv2d(g_t, gauss_filter, padding=kernel_size//2, groups=1)
    sigma_sq_f = F.conv2d(pred*pred, gauss_filter, padding=kernel_size//2, groups=1) - mu_f.pow(2)
    sigma_sq_y = F.conv2d(g_t*g_t, gauss_filter, padding=kernel_size//2, groups=1) - mu_y.pow(2)
    sigma_sq_fy = F.conv2d(pred*g_t, gauss_filter, padding=kernel_size//2, groups=1) - mu_f*mu_y
    c_1 = 0.01**2
    c_2 = 0.03**2
    ssim_val = ((2 * mu_f * mu_y + c_1) * (2 * sigma_sq_fy + c_2)) / ((mu_f.pow(2) + mu_y.pow(2) + c_1) * (sigma_sq_f + sigma_sq_y + c_2))
    return ssim_val.mean()


# implementation of SSIM loss

class SSIM_Loss(nn.Module):
    def __init__(self, kernel_size=11):
        super(SSIM_Loss, self).__init__()
        self.kernel_size = kernel_size
        self.gauss_filter = gaussian_filter(1, kernel_size, 1.5)

    def forward(self, pred, g_t):
        self.gauss_filter = self.gauss_filter.cuda(pred.get_device()) if pred.is_cuda else self.gauss_filter
        self.gauss_filter = self.gauss_filter.type_as(pred) if not self.gauss_filter.type() == pred.data.type() else self.gauss_filter
        mse_loss = torch.mean((pred - g_t)**2, dim=(0, 1, 2, 3))
        ssim_loss = 1 - ssim_index(pred, g_t, self.gauss_filter, self.kernel_size)
        return torch.mul(torch.add(torch.mul(ssim_loss, 0.001), mse_loss), pred.shape[2] * pred.shape[3])


