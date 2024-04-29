__author__ = "Jaideep Murkute"

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from focal_loss import FocalLoss


args = argparse.Namespace()
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.cuda.set_device(0)
device = torch.device("cuda" if cuda_available else "cpu")
args.device = device


plt.figure()
colors = ['green', 'orange', 'red', 'brown', 'black']
gammas = [0, 0.5, 1, 2, 5]
for i, gamma in enumerate(gammas):
    gamma = torch.Tensor([gamma]).to(args.device)

    target = []
    all_probs = []
    for prob in np.arange(0.01, 1, 0.01):
        all_probs.append(torch.Tensor([prob]))
        target.append(torch.Tensor([1.0]))

    target = torch.cat(target).to(args.device)
    all_probs = torch.cat(all_probs).to(args.device)

    target = target.view(-1, 1)
    all_probs = all_probs.view(-1, 1)
    loss = FocalLoss(args, gamma=gamma, logits=False)(all_probs, target)
    loss = loss.cpu().numpy()
    plt.plot(loss, color=colors[i], label='gamma: {}'.format(str(gamma)))

plt.legend()
plt.show()
plt.close()