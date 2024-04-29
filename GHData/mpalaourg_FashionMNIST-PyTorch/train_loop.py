import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from Network import Network
from RunManager import RunManager
from RunBuilder import RunBuilder

import warnings
from collections import OrderedDict

torch.set_printoptions(linewidth=120)
warnings.filterwarnings(action="ignore", category=UserWarning, message="CUDA initialization.*")
# torchvision gives access to: Datasets, Models, Transforms, Utils
# For PyTorch, FashionMNIST and MNIST, the difference is only one url. [FashionMNIST extends MNIST]
'''
    1. MNIST extends VisionDataset which extends data.Dataset
    2. Domain name of data being fetched: http://yann.lecun.com/exdb/mnist
    3. Yann LeCun is Director of AI Research at Facebook (and more) and has invented CNNs
'''

if __name__ == "__main__":
    train_set = torchvision.datasets.FashionMNIST(         # get instance of the FashionMNIST dataset
        root='./data/FashionMNIST',     #
        train=True,                     # Extract
        download=True,                  #
        transform=transforms.Compose([
            transforms.ToTensor()      # Transform
        ])
    )
    # Calculate mean and std
    loader = DataLoader(train_set, batch_size=len(train_set))
    data = next(iter(loader))
    mean, std = data[0].mean(), data[0].std()

    train_set_normal = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )
    # Dictionary to keep both of train sets
    trainsets = {
        'not_normal': train_set,
        'normal': train_set_normal
    }

    params = OrderedDict(
        # batch_size=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        # lr=[.01, .001, .0001, .00001]
        batch_size=[10, 20, 50, 100],
        lr=[.01, .001],
        trainset=['not_normal', 'normal']
    )

    m = RunManager()
    for run in RunBuilder.get_runs(params):
        network = Network()  # create instance of the my neural network
        loader = DataLoader(
            trainsets[run.trainset],
            batch_size=run.batch_size,
            shuffle=True
        )                    # Load   |   wraps the dataset inside the dataloader
        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        m.begin_run(run, network, loader)
        for epoch in range(20):
            m.begin_epoch()
            for batch in loader:
                images, labels = batch

                preds = network(images)
                loss = F.cross_entropy(preds, labels)  # Calculating the loss
                optimizer.zero_grad()                  # Start over for the gradients
                loss.backward()                        # Calculating the gradients
                optimizer.step()                       # Updating the Weights

                m.track_loss(loss, batch)
                m.track_num_correct(preds, labels)
            m.end_epoch()
        m.end_run()
    m.save('results')
