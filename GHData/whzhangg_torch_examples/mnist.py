from mlp import ConvNet
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms

from ray import tune
import numpy as np

def main(
    batch_size: int = 64,
    lr = 1e-4,
    epochs = 10,
    verbose = True
):
    test_batch_size = 256

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('/home/wenhao/python/torch-examples/data', train=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/home/wenhao/python/torch-examples/data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ConvNet()
    print(model)
    optimizer = torch.optim.Adadelta(model.parameters(), lr= lr)

    for epoch in range(1, epochs + 1):
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        if verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    """
    step_size = 256
    steps = 0
    tot_step = 300
    while steps < tot_step:
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1)*batch_size % step_size == 0:
                # we do a step here
                steps += 1
                if verbose:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        output = model(data)
                        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(test_loader.dataset)

                if verbose:
                    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                        test_loss, correct, len(test_loader.dataset),
                        100. * correct / len(test_loader.dataset)))
            
            if steps > tot_step:
                break

    print(steps)
    """
                
def ray_targetfunc(config, print_result:bool = False):

    step_size = 2048
    test_batchsize = 1000

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('/home/wenhao/python/torch-examples/data', train=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/home/wenhao/python/torch-examples/data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size = config["batchsize"])
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size = test_batchsize)

    model = ConvNet(
        nconvolution=config["nconvolution"],
        initial_channel=config["initial_channel"],
        dropout1=config["dropout1"],
        dropout2=config["dropout2"],
        n_linear=config["n_linear"]
    )
    optimizer = torch.optim.Adadelta(model.parameters(), lr= config['lr'])

    ntest = 2000

    for epoch in range(1, config['epoch'] + 1):
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1)*config['batchsize'] % step_size == 0:
                # we do a step here
            
                test_loss = 0
                correct = 0
                with torch.no_grad():

                    for batch_id_test, (data, target) in enumerate(test_loader):
                        output = model(data)
                        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        if (batch_id_test+1) * test_batchsize >= ntest:
                            break

                test_loss /= len(test_loader.dataset)

                if print_result:
                    print('Train Epoch: {:>2d} [{:>6d}/{:>6d} ({:2.0f}%)]  Accuracy: {:>6d}/{:>6d} ({:2.0f}%)'.format(
                        epoch, (batch_idx + 1)*config['batchsize'], 
                        len(train_loader.dataset),
                        100 * (batch_idx + 1)*config['batchsize'] / len(train_loader.dataset), 
                        correct, 
                        ntest,
                        100. * correct / ntest
                        )
                    )
                else:
                    tune.report(loss = test_loss, accuracy = 100. * correct / ntest)
    return

def dummy_target(config):
    tune.report(np.random.randint(10))

def ray_optimize():
    config = {
        "batchsize": tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
        "lr": tune.qloguniform(1e-5, 1, 1e-5),
        "epoch": 4,
        "nconvolution": tune.choice([1, 2, 3]),
        "initial_channel": tune.choice([4, 8, 16, 32]),
        "dropout1": 0.25, 
        "dropout2": 0.5, 
        "n_linear":tune.sample_from(lambda _: 2**np.random.randint(4, 9))
    }
    scheduler_ASHA = tune.schedulers.ASHAScheduler(
                    metric="accuracy",
                    mode="max",
                    max_t=120,
                    grace_period=2
                )

    scheduler_median = tune.schedulers.MedianStoppingRule(
                    metric="accuracy",
                    mode="max",
                    grace_period=10
    )

    scheduler_PBT = tune.schedulers.PopulationBasedTraining(
                    metric='accuracy',
                    mode='max',
                    perturbation_interval=5,
                    hyperparam_mutations={
                        "batchsize": tune.choice([8,16,32,64]),
                        "lr": tune.qloguniform(1e-5, 1, 1e-5),
                        #"nconvolution": tune.choice([1, 2, 3]),
                        "initial_channel": tune.choice([4, 8, 16, 32]),
                        "n_linear": tune.choice([8,16,32,64,128,256])
                    }
    )

    result = tune.run(
        ray_targetfunc, 
        config = config, 
        local_dir = 'ray_mnist',
        #num_samples= 20,
        num_samples = 10,
        scheduler=scheduler_PBT,
        name = 'convolution_PBT',
        #sync_config=tune.SyncConfig(syncer = None)  # Disable syncing
    )


    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))

def analysis_result():
    # folders are ray_mnist/convolution/ray_targetfunc_43efb_00000 ~ 00009
    import os
    from DFTtools.tools import read_json

    import json
    from io import StringIO
    root = "ray_mnist/convolution_PBT"
    folders = os.listdir(root)
    folders = sorted([ f for f in folders if ("97fff" in f and ".txt" not in f) ])
    all_accuracies = []
    for folder in folders:
        parameter = read_json(os.path.join(root, folder, "params.json"))
        with open(os.path.join(root, folder, "result.json"),'r') as f:
            lines = f.readlines()
        
        iterations = [ json.load(StringIO(line)) for line in lines ]
        accuracies = [ iterat['accuracy'] for iterat in iterations  ]
        all_accuracies.append(accuracies)

    from mplconfig import get_acsparameter
    import matplotlib.pyplot as plt

    filename = "PBT.pdf"
    with plt.rc_context(get_acsparameter(width = "single", n = 1, color = "line")):
        fig = plt.figure()
        axes = fig.subplots()
                
        axes.set_xlabel("Time (1 step = 2048 samples)")
        axes.set_ylabel("Accuracy (%)")
        
        axes.set_xlim(1, 120)
        axes.set_ylim(8, 100)
        
        for i, acc in enumerate(all_accuracies):
            axes.plot(range(1, len(acc)+1), acc, '-', label = "{:0>2d}".format(i+1))
            
        axes.legend(ncol = 4)

        fig.savefig(filename)

def fixed_parameter():
    config = {
        "batchsize": 64,
        "lr": 0.1,
        "epoch": 4,
        "nconvolution": 2,
        "initial_channel": 32,
        "dropout1": 0.25, 
        "dropout2": 0.5, 
        "n_linear": 128
    }
    ray_targetfunc(config, print_result = True)


if __name__ == "__main__":
    analysis_result()