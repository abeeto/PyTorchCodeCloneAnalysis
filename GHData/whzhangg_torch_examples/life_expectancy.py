import torch
from torch import nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np

from mlp import MultiLinear
from data import get_life_expectance_data


def train_loop(n_hidden, depth, lr:float, cycle = 100):
    trainloader, testloader, (nfeature, ntarget) = get_life_expectance_data()
    loss_fn = nn.MSELoss()
    middle = tuple([n_hidden] * depth)
    network = MultiLinear(nfeature, ntarget, middle)
    optimizer =torch.optim.AdamW(network.parameters(), lr=lr)

    for i in range(cycle):
        for x, y in trainloader:
            optimizer.zero_grad()
            pred = network(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        error = 0
        count = 0
        with torch.no_grad():
            for x, y in testloader:
                pred = network(x)
                error += loss_fn(pred, y) 
                count += y.shape[0]
        print("epoch: {:>5d}       ------------ MSE: {:>10.4e}".format(i+1, error/count))


def train_ray(config, print_result:bool = False):
    n_hidden = config['nhidden']
    depth = config['depth']
    learning_rate = config['lr']
    ncycle = config['circle']
    batchsize = config['batchsize']

    trainloader, testloader, (nfeature, ntarget) = get_life_expectance_data(batchsize=batchsize)
    loss_fn = nn.MSELoss()
    middle = tuple([n_hidden] * depth)
    network = MultiLinear(nfeature, ntarget, middle)
    optimizer =torch.optim.AdamW(network.parameters(), lr=learning_rate)

    for epoch in range(ncycle):
        for x, y in trainloader:
            optimizer.zero_grad()
            pred = network(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        error = 0
        count = 0
        with torch.no_grad():
            for x, y in testloader:
                pred = network(x)
                error += loss_fn(pred, y) 
                count += y.shape[0]

        if print_result:
            print("epoch: {:>5d}       ------------ MSE: {:>10.4e}".format(epoch+1, error/count))
 
        tune.report(loss = (error/count).item())  # we need to pass a scalar 
        

def main():

    config = {
        "nhidden": tune.choice([10, 20, 30, 40]),
        "depth": tune.choice([3,4,5]),
        "lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batchsize": tune.choice([8, 16, 32]),
        "circle": 100
    }

    scheduler_median = tune.schedulers.MedianStoppingRule(
                    metric="loss",
                    mode="min",
                    min_samples_required = 8,
                    grace_period=10
    )

    scheduler_PBT = tune.schedulers.PopulationBasedTraining(
                    metric='loss',
                    mode='min',
                    perturbation_interval=5,
                    hyperparam_mutations={
                        "nhidden": tune.choice([10, 20, 30, 40]),
                        "depth": tune.choice([3,4,5]),
                        "lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
                        "batchsize": tune.choice([8, 16, 32]),
                    }
    )
                
    result = tune.run(
        train_ray,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=8,
        scheduler=scheduler_PBT,
        local_dir = 'ray_life',
        name = 'life_exp'
    )



    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
                                
                                
if __name__ == "__main__":
    main()
