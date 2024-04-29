from __future__ import division
import numpy as np
import os
import torch
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def make_3d_grid(b_min, b_max, resolution):
    ''' Makes a 3D grid.
    Args:

    '''
    print(resolution)

    batch = 1
    point =  torch.empty((resolution+1) * (resolution+1) * (resolution+1), 3, 1).to(device)
    point1 =  torch.empty((resolution+1) * (resolution+1) * (resolution+1), 3, 1).to(device)

    s = 0

    for i in range(resolution+1):
        print(i)
        for j in range(resolution+1):
            for k in range(resolution+1):
                #print(s)
                #point.append([i,j,k])
                point[s, :, :] = torch.FloatTensor([[i],[j],[k]]).float()

                gridx = (b_max[0]-b_min[0])/resolution
                gridy = (b_max[1]-b_min[1])/resolution
                gridz = (b_max[2]-b_min[2])/resolution


                px = i*gridx+b_min[0]
                py = j*gridy+b_min[1]
                pz = k*gridz+b_min[2]

                point1[s, :, :] = torch.FloatTensor([[px],[py],[pz]]).float()

                s = s+1

    return point, point1

def make_3d_grid1(b_min, b_max, resolution):
    ''' Makes a 3D grid.
    Args:

    '''
    print(resolution)
    #point = []
    #point1 = []
    batch = 1
    point =  torch.empty(batch, 3, 1, 1, (resolution+1) * (resolution+1) * (resolution+1)).to(device)
    point1 =  torch.empty(batch, 3, 1, 1, (resolution+1) * (resolution+1) * (resolution+1)).to(device)

    s = 0
    for i in range(resolution+1):
        for j in range(resolution+1):
            for k in range(resolution+1):
                #print(s)
                #point.append([i,j,k])
                point[0, :, :, :, s] = torch.FloatTensor([[[i]],[[j]],[[k]]]).float()

                gridx = (b_max[0]-b_min[0])/resolution
                gridy = (b_max[1]-b_min[1])/resolution
                gridz = (b_max[2]-b_min[2])/resolution


                px = i*gridx+b_min[0]
                py = j*gridy+b_min[1]
                pz = k*gridz+b_min[2]

                point1[0, :, :, :, s] = torch.FloatTensor([[[px]],[[py]],[[pz]]]).float()

                s = s+1


    return point, point1

#p, pp = make_3d_grid([0,0,0], [2,2,2], 4)
#print(pp)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint='checkpoints/', filename='checkpoint.pth.tar', snapshot=None):
    """Saves checkpoint to disk"""
    # todo: also save the actual preds
    # preds = to_numpy(preds)
    #if not os.path.exists(checkpoint):
    #    os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    '''
    if snapshot and state.epoch % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    '''

# check the number of parameters of a model?
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transfer_optimizer_to_gpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        if tensor.requires_grad:
            return tensor.detach().numpy()
        else:
            return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def binary_pred_accuracy(preds, labels):
    N = preds.shape[0]
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    correct = preds == labels
    num_correct = np.sum(correct)
    return num_correct * 1.0 / N
