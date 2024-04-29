import torch
import torch.nn.functional as F
from options import Options

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = Options().parse()

def gradient_magnitude(d_batch, n_pixels):
    a = torch.Tensor([[[[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]]])
    b = torch.Tensor([[[[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]]])
    a = a.to(device)
    b = b.to(device)
    G_x = F.conv2d(d_batch, a, padding=1).to(device)
    G_y = F.conv2d(d_batch, b, padding=1).to(device)
    G = torch.pow(G_x, 2) + torch.pow(G_y, 2)
    return G

#pred.shape        -> [16, 1, 120, 160]
#truth.shape -> [16, 120, 160]

def gradient_loss(pred, truth):
    n_pixels = truth.shape[-2] * truth.shape[-1]
    truth = truth.unsqueeze(1)
    dif = torch.abs(gradient_magnitude(pred, n_pixels)-gradient_magnitude(truth, n_pixels))
    dif = dif.view(-1,n_pixels)
    return dif.sum(dim=1).mean()

def scale_invariant_loss(pred, truth):
    n_pixels = truth.shape[1] * truth.shape[2]
    pred = (pred * 0.225) + 0.45
    pred = pred * 255
    pred[pred <= 0] = 0.00001
    truth[truth == 0] = 0.00001
    truth.unsqueeze_(dim=1)
    d = torch.log(pred) - torch.log(truth)
    term_1 = torch.pow(d.view(-1, n_pixels), 2).mean(dim=1).sum()  # pixel wise mean, then batch sum
    term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1), 2) / (2 * (n_pixels ** 2))).sum()
    return term_1 - term_2

def rmse_loss(pred, truth):
    pred[pred <= 0] = 0.00001
    truth[truth == 0] = 0.00001
    truth = truth.view(pred.size())
    return torch.sqrt(torch.mean((pred - truth) ** 2))

def logrmse_loss(pred, truth):
    pred[pred <= 0] = 0.00001
    truth[truth == 0] = 0.00001
    truth = truth.view(pred.size())
    return torch.sqrt(torch.mean((torch.log(pred) - torch.log(truth)) ** 2))

def all_loss(pred,truth):
    eigen = scale_invariant_loss(pred,truth)
    rmse = rmse_loss(pred,truth)
    logrmse = logrmse_loss(pred, truth)
    if opt.image_gradient:
        grad = gradient_loss(pred,truth)
        return eigen, rmse, logrmse,grad
    return eigen, rmse, logrmse
