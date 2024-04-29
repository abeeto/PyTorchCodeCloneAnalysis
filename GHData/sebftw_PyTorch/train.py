import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time, os
from sklearn.model_selection import KFold
import math
from multiprocessing import Pool


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, time2-time1))
        return ret
    return wrap

# 8 layers of 128 neurons each, mini-batches of 64.

class MNISTnet2(torch.nn.Module):
    def __init__(self):
        super().__init__() # Same as super(MNISTnet2, this).__init__().
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.fullc = torch.nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(),
            nn.Linear(500, 10))

    def forward(self, x):
        x = self.conv(x)
        x = self.fullc(x.view(-1, 4 * 4 * 50))
        return x


class MNISTnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.hidden = nn.Sequential(*[nn.ReLU(), nn.Dropout(0.5), nn.ReLU()] )
        scale = 8
        hidden_layers = [nn.Linear(28*28, 128 * scale)]
        for _ in range(3):
            hidden_layers.append(nn.Linear(128 * scale, 128 * scale))
            hidden_layers.append(nn.SELU())  # SELU faster than ELU?
            hidden_layers.append(nn.AlphaDropout(0.1))
            #hidden_layers.append(nn.Dropout(0.1))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(128 * scale, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.output(x)
        return x


def profile(func, use_cuda=torch.cuda.is_available(), path=None):
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        func()
    if path is not None:
        prof.export_chrome_trace(path)
    return prof
    # with torch.cuda.profiler.profile():
    #    model(train_set[0])  # Warmup CUDA memory allocator and profiler
    #    with torch.autograd.profiler.emit_nvtx():




class PrePreprocess(torch.utils.data.Dataset):
    # This class saves datasets (after they're transformed/preprocessed) to desk
    # This may take up a lot of disk space, but it saves some seconds everytime
    #  on every run of the script. Makes it faster to debug/test.
    @staticmethod
    def preprocess(dataset):
        # Just try to max out all CPUs
        # If we get an out of memory error, batch_size is too large.
        num_workers, batch_size = os.cpu_count(), 1
        if num_workers is None:
            num_workers = 0
        else:
            batch_size = int(len(dataset) / num_workers)

        # If dataset is extremely large, a batch of this size may not fit in memory.
        loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
        data, target = [], []
        for result in loader:
            data.extend(result[0])
            target.extend(result[1])
        return list(zip(data, target))

    def __init__(self, dataset, path=None, *args, **kwargs):
        if callable(dataset):
            dataset = dataset(*args, **kwargs)

        if path is not None:
            try:
                self.memo = torch.load(open(path, "rb"))
            except (OSError, IOError) as _:
                print("Failed to load pre-preprocessed data. Creating files.")
                self.memo = PrePreprocess.preprocess(dataset)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.memo, open(path, "wb"))
        else:
            self.memo = PrePreprocess.preprocess(dataset)

    def __getitem__(self, index):
        return self.memo[index]

    def __len__(self):
        return len(self.memo)


class TrainSupervisor(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        super().step(self, *args, **kwargs)


class Trainer(object):
    @timing
    def train_epoch(self, model, train_loader, optimizer, device):
        model.train()
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=self.non_blocking)
            target = target.to(device, non_blocking=self.non_blocking)
            optimizer.zero_grad()
            output = model(data)
            loss = self.loss_fn(output, target)
            train_losses.append(loss)
            self.reduction_fn(loss).backward()  # get average (or other reduction) over batch
            optimizer.step()
        return torch.cat(train_losses)

    @timing
    def test_epoch(self, model, test_loader, device):
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=self.non_blocking)
                target = target.to(device, non_blocking=self.non_blocking)
                output = model(data)
                loss = self.loss_fn(output, target)
                test_losses.append(loss)
        return torch.cat(test_losses)

    @staticmethod # can be used as loss_fn together with (default) mean reduction, if accuracy is of interest.
    def class_loss_fn(output, target):
        """ NOT to be used for training - only validation! """
        # Returns the loss in addition to the error on every sample.
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        return pred.eq(target.view_as(pred))

    def __init__(self, train_set, test_set,
                 loss_fn=lambda x, y: F.cross_entropy(x,y,reduction='none'),
                 reduction_fn=lambda loss: loss.mean(), non_blocking=True, pin_memory=False):
        # try pin_memory if transfer to GPU is bottleneck
        self.loss_fn = loss_fn
        self.reduction_fn = reduction_fn
        self.non_blocking = non_blocking
        self.pin_memory = pin_memory
        self.train_set = train_set
        self.val_set = test_set

    @staticmethod
    def find_max_batch(model, dataset, loss_fn, reduction_fn, device):
        # This function should return a batch size that guarantees that there will be no
        #  RuntimeError when training with that size. This size should be as large as possible.

        # If dataset is extremely large, this may run out of (RAM) memory:
        data_set, target_set = zip(*dataset)
        data_set, target_set = torch.stack(data_set), torch.stack(target_set)
        losses = []
        n = 0
        while 1 << n <= len(dataset):
            try:
                data = data_set[:(1 << n), ...].to(device, non_blocking=True)
                target = target_set[:(1 << n), ...].to(device, non_blocking=True)
                output = model(data)
                loss = loss_fn(output, target)
                reduction_fn(loss).backward()

                losses.append(loss)
                reduction_fn(torch.cat(losses)).item()

                n += 1
            except RuntimeError:
                if n == 0:
                    # If we could not even have one sample on the GPU.
                    raise
                else:
                    break

        if 0 < n:  # Be conservative (in case we are at the limit).
            n -= 1

        return 1 << n

    def train(self, model, batch_size, optimizer, scheduler=None, max_epochs=10,
              patience=10, rel_threshold=0.01, max_batch_size=None):
        # optimizer.step is called on every mini-batch.
        # scheduler.stop is called on every epoch, given val_loss and epoch

        # Later we can pass a train-supervisor as argument. This class should have a function to register
        #  each iteration of training. And it should return whether the training should break early.
        #  it should be given train_losses, val_losses, model, optimizer, batch_size so it can checkpoint all this.
        #  this mid-train checkpoint is only if model takes days to train. Otherwise it should just be saved when
        #  the training is done, and this function returns.
        #   The training supervisor could also stop training after a specific time has passed!
        device = next(model.parameters()).device  # Put tensors on same device as model. (Could also be arg)
        # Find maximal batch size.
        if max_batch_size is None:
            dataset = self.train_set if len(self.train_set) >= len(self.val_set) else self.val_set
            max_batch_size = Trainer.find_max_batch(model, dataset, self.loss_fn, self.reduction_fn, device)

        # Start by finding largest possible batch size.
        # Simply save seed, instead of weights, so we can re-do the training.
        # Should be given a test and validation set (generated for K-fold cross-validation)
        #  - and the training set should then be randomized.
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                                   pin_memory=self.pin_memory)
        test_loader = torch.utils.data.DataLoader(self.val_set, batch_size=max_batch_size,
                                                  pin_memory=self.pin_memory)

        best_val_loss = None
        num_bad_epochs = 0
        train_losses, val_losses = [], []
        for epoch in range(max_epochs):
            train_loss = self.reduction_fn(self.train_epoch(model, train_loader, optimizer, device)).item()
            val_loss = self.reduction_fn(self.test_epoch(model, test_loader, device)).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if scheduler is not None:
                try:
                    should_stop = scheduler.step(val_loss)
                except TypeError:
                    should_stop = scheduler.step()
                if should_stop is not None and should_stop:
                    break

            # Move the logic below here into the scheduler.
            if best_val_loss is None or val_loss < best_val_loss * (1 - rel_threshold):
                best_val_loss = val_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= patience:
                    print("Early stopping")
                    break
        else:
            num_bad_epochs = 0  # late stopping, haven't hit plateau.
            print("Late stopping.")
        epochs = epoch - num_bad_epochs  # We do not care for after the plateau.
        print("Training done. Best:", min(train_losses), min(val_losses))
        return train_losses[:epochs], val_losses[:epochs]

    @staticmethod
    def plot_training(train_losses, val_losses):
        offset = 10
        epochs = len(train_losses)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        # ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        color = 'tab:red'
        ax1.plot(range(offset, epochs), train_losses[offset:], '-',
                 range(offset, epochs), val_losses[offset:], '--', color=color)
        ax1.legend(['training', 'validation'])
        ax1.tick_params(axis='y', labelcolor=color)
        # Training seems to follow a power law curve: https://en.wikipedia.org/wiki/Learning_curve

        # Additionally we could plot more statistics: like how concave the function is, moving averages, etc.
        # ax2 = ax1.twinx()
        # color = 'tab:blue'
        # ax2.set_ylabel('acc')  # we already handled the x-label with ax1
        #  ax2.plot(epoch_range, [-math.log(1/x-1) for x in test_accs[offset:epochs]], '--', color=color)
        # ax2.plot(range(epochs), test_accs[:epochs], '--', color=color)
        # ax2.plot([offset + math.log(x/offset)/math.log(epochs/offset) * (epochs-offset) for x in range(offset, epochs)],
        #         [math.log(x) for x in val_losses[offset:]], '--', color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        # ax2.invert_yaxis()
        # We want statistics for the lowest/highest weight, also lowest absolute.

        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
        ax2.loglog(range(offset, epochs), train_losses[offset:], '-',
                   range(offset, epochs), val_losses[offset:], '--', color=color)
        ax2.legend(['training', 'validation'])
        ax1.tick_params(axis='y', labelcolor=color)
        plt.grid(True)

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    torch.manual_seed(4)  # reproducible, since we shuffle in DataLoader.

    # Load and and preprocess datasets (or load pre-preprocessed)
    data_root = 'mnist'
    train_set = PrePreprocess(torchvision.datasets.MNIST,
                              os.path.join(data_root, 'preprocessed', 'training.pt'),
                              data_root, train=True, download=True, transform=transforms.ToTensor())
    test_set = PrePreprocess(torchvision.datasets.MNIST,
                             os.path.join(data_root, 'preprocessed', 'test.pt'),
                             data_root, train=False, download=True, transform=transforms.ToTensor())

    # For test loader, the batch_size should just be as large as can fit in memory.
    # train_batch_size = 256 * 3
    # test_batch_size = 2 << 9

    # Try getting gradient over training set, and then scale by size of gradient over test set. / should not work

    # Batch size should be as large a possible (but limited by memory) for speed
    #  Then to (hyper) optimize it should be halved until that does not improve scores.
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, pin_memory=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model = MNISTnet().to(device, non_blocking=True)
    #optimizer = optim.Adagrad(model.parameters(), lr=0.1)
    #Trainer(train_set, test_set).train(model, optimizer, batch_size=128*2, max_epochs=200)
    split_point = int(len(train_set) * 5/6)
    train_set, val_set = torch.utils.data.random_split(train_set, (split_point, len(train_set) - split_point))
    trainer = Trainer(train_set, val_set)
    for lr_i in [1, 0.1]:
        # If model is extremely large, this may run out of memory.
        model1 = MNISTnet2().to(device, non_blocking=True)
        optimizer = optim.Adagrad(model1.parameters(), lr=lr_i)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=0.1, verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        train_result = trainer.train(model1, 128*2*2, optimizer, scheduler, max_epochs=200, patience=10)
        Trainer.plot_training(*train_result)
    #print("Number of parameters:", list(map(len, list(model.parameters()))))
    # train(model, train_loader, optimizer, device=device)
    # test_loss, test_acc = test(model, train_loader, device=device)
    # print(test_acc)
    # We need to optimize hyper parameters. Also a way of stopping early.


