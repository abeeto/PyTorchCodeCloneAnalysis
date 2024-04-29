import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import argparse
import pathlib

class CA(nn.Module):
    def __init__(self, ca_size=(32, 32), n_channels=16, fire_rate=0.5, living_threshold=0.1):
        super(CA, self).__init__()
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.ca_size = ca_size
        self.living_threshold = living_threshold
        self.dx = torch.from_numpy(np.asarray([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])/8.).view(1,1,3,3).repeat(n_channels, n_channels, 1, 1).float()
        self.dy = torch.from_numpy(np.asarray([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]).T/8.).view(1,1,3,3).repeat(n_channels, n_channels, 1, 1).float()
        self.conv1 = nn.Conv2d(48, 128, 1)
        self.conv2 = nn.Conv2d(128, n_channels, 1)
        self.conv2.weight.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def get_living(self, x):
        alpha = x[:, 3, :, :]
        return nn.functional.max_pool2d(alpha, 3, 1, 1) > self.living_threshold

    def forward(self, x):
        pre_update = self.get_living(x)
        dx = nn.functional.conv2d(x, self.dx.to(next(self.parameters()).device), padding=1)
        dy = nn.functional.conv2d(x, self.dy.to(next(self.parameters()).device), padding=1)
        y = torch.cat((x, dx, dy), 1)
        y = self.conv1(y)
        y = torch.nn.functional.leaky_relu(y)
        y = self.conv2(y)

        rand_mask = (torch.rand(x.size(0), *self.ca_size) <= self.fire_rate).unsqueeze(1).repeat(1, x.size(1), 1, 1).to(next(self.parameters()).device)
        x = x+y*rand_mask
        post_update = self.get_living(x)
        alive = (pre_update*post_update).unsqueeze(1).repeat(1, x.size(1), 1, 1)
        return x*alive

def train(target, size_batch=4, size_pool=32, img_size=32, progress_path=pathlib.Path('progress'), model_path=pathlib.Path('model'), device='auto'):
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    progress_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    model = CA((img_size, img_size)).to(device)
    orig_state = np.zeros((1, 16, img_size, img_size))
    orig_state[:, 3:, img_size // 2, img_size // 2] = 1.0
    plt.imshow(orig_state[0, :3, :, :].transpose((1, 2, 0)))
    plt.savefig('seed.png')
    plt.close()
    orig_state = torch.from_numpy(orig_state).float().to(device)

    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=500)

    img_np = Image.open(target).resize((img_size, img_size))
    img_np = np.array(img_np) / 255.

    plt.imshow(img_np[:, :, :3])
    plt.savefig('target.png')
    plt.close()
    img = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0).float().to(device).repeat(size_batch, 1, 1, 1)

    pool = orig_state.detach().cpu().repeat(size_pool, 1, 1, 1)
    n_epochs = 8000
    local_min = np.inf
    for epoch in range(n_epochs):
        optim.zero_grad()

        perm = torch.randperm(pool.size(0))
        idx = perm[:size_batch]
        state = pool[idx]
        state = state.to(device)
        worst_idx = torch.topk(
            torch.mean(torch.nn.functional.mse_loss(state[:, :4, :, :], img[:, :4, :, :], reduction='none'),
                       dim=(1, 2, 3)), k=2).indices.tolist()
        for i, j in enumerate(worst_idx):
            state[j] = state[i]
            state[i] = orig_state.detach()

        for _ in range(np.random.randint(64, 96)):
            state = model(state)

        loss = torch.nn.functional.mse_loss(state[:, :4, :, :], img[:, :4, :, :])

        loss.backward()
        for p in model.parameters():
            p.grad = p.grad / (torch.norm(p.grad) + 1e-8)
        optim.step()
        scheduler.step(loss)

        local_min = min(local_min, loss.item())

        sys.stdout.write('\rEpoch: {}\tlog10-Loss: {:.4}\tlog10-min: {:.4}'.format(epoch, np.log10(loss.item()),
                                                                                   np.log10(local_min)))
        sys.stdout.flush()

        if epoch % 10 == 0:
            img_new = state.detach().cpu().numpy()[:, :3, :, :].reshape(size_batch, 3, img_size, img_size).transpose(
                (0, 2, 3, 1)).clip(0., 1.0)
            fig = plt.figure(figsize=(8, 8))
            for i in range(1, int(np.sqrt(size_batch)) ** 2 + 1):
                fig.add_subplot(int(np.sqrt(size_batch)), int(np.sqrt(size_batch)), i)
                plt.imshow(img_new[i - 1])
            plt.tight_layout()
            plt.savefig('progress.png')
            plt.savefig(progress_path/'{}.png'.format(epoch))
            plt.close()

        if (epoch + 1) % 100 == 0:
            torch.save(model, model_path/'{}.pt'.format(epoch))

        pool[idx] = state.detach().cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Neural Cellular Automata to grow into the shape of a target image.')
    parser.add_argument('target', type=pathlib.Path, help='target image')
    parser.add_argument('--batchsize', '-bs', type=int, default=4, help='batch size')
    parser.add_argument('--poolsize', '-ps', type=int, default=32, help='pool size')
    parser.add_argument('--casize', '-cas', type=int, default=32,
                        help='CA size')
    parser.add_argument('--progresspath', '-pp', type=pathlib.Path, default=pathlib.Path('progress'), help='path to save progress visualisations')
    parser.add_argument('--modelpath', '-mp', type=pathlib.Path, default=pathlib.Path('model'), help='path to save models')
    parser.add_argument('--device', '-d', type=str, default='auto',
                        help='device to use for training (auto, cuda or cpu)')

    args = parser.parse_args()
    train(args.target, size_batch=args.batchsize, size_pool=args.poolsize, img_size=args.casize, progress_path=args.progresspath, model_path=args.modelpath, device=args.device)
