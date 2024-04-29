
import os
import torch
from network import *
from scope import scope
from optim import Optim
from torchvision import datasets, transforms

with scope("Variables Definition"):
    batch_size = 100
    traning_ratio = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

with scope("Data Loader"):
    loader = {}
    for stage in ("train", "test"):
        more = dict(num_workers = 1, pin_memory = True) if torch.cuda.is_available() else {}
        mnist = datasets.MNIST(os.curdir, train = stage == "train", download = True, transform = transforms.ToTensor())
        loader[stage] = torch.utils.data.DataLoader(mnist, batch_size = batch_size, shuffle = stage == "train", **more)

with scope("Models Construction"):
    G = Generator()
    D = Discriminator()

with scope("Optimizer Builder"):
    hyper = dict(lr = lambda i: 0.1 / (1.00004) ** i, momentum = lambda i: 0.5 + min(0.2, i / 1e6))
    Gopt_generator = Optim(torch.optim.SGD, G.parameters(), **hyper)
    Dopt_generator = Optim(torch.optim.SGD, D.parameters(), **hyper)

with scope("Optimization Step"):
    def run(iteration, model, entry, is_training = True):
        if is_training: eval(model).train()
        else: eval(model).eval()

        x, y = entry
        n_batch = x.size(0)
        x_flat = x.flatten(1)
        y_hot = torch.zeros(n_batch, 10)
        y_hot[y] = 1
        z = torch.rand(n_batch, 100)

        loss = torch.log(D(x_flat, y_hot)) + torch.log(1 - D(G(z, y_hot), y_hot))
        loss = loss.mean()
        opt = next(eval(model + "opt_generator"))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("iteration:", iteration, "|", "loss:", loss.item())
        if opt.state_dict()['param_groups'][0]['lr'] < 1e-6: raise RuntimeError("JUMP")


with scope("Main Loop"):
    iteration = 1
    while True:
        for x in loader['train']:
            if iteration % (traning_ratio) == 0: run(iteration, 'G', x)
            else: run(iteration, 'D', x)
            iteration += 1

print("DONE")
