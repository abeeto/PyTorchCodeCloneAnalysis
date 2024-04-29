import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from Network import Network
from RunBuilder import RunBuilder

from collections import OrderedDict
from utils import get_all_preds, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, message="CUDA initialization.*")
""" best for batch_size = 10, lr = 0.001 , take some time extra (initial run)"""
""" best for batch_size = 20, lr = 0.001 , normalized """

if __name__ == "__main__":
    # Calculate mean and std on train data, then hardcoded for my ease
    mean, std = torch.tensor(0.28604060411453247), torch.tensor(0.3530242443084717)

    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )
    test_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )
    # Final training and then predict on test data
    params = OrderedDict(
        batch_size=[20],
        lr=[.001]
    )
    for run in RunBuilder.get_runs(params):
        network = Network()
        train_loader = DataLoader(
            train_set,
            batch_size=run.batch_size,
            shuffle=True
        )
        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        for epoch in range(20):
            for batch in train_loader:
                images, labels = batch

                preds = network(images)
                loss = F.cross_entropy(preds, labels)  # Calculating the loss
                optimizer.zero_grad()                  # Start over for the gradients
                loss.backward()                        # Calculating the gradients
                optimizer.step()                       # Updating the Weights

    # Here the network is already trained and will be tested on test data
    plt.figure()
    test_loader = DataLoader(test_set, batch_size=1000)
    test_preds = get_all_preds(network, test_loader)
    cm = confusion_matrix(test_set.targets, test_preds.argmax(dim=1))
    plot_confusion_matrix(cm, test_set.classes)
    plt.show()

    train_loader = DataLoader(train_set, batch_size=1000)
    train_preds = get_all_preds(network, train_loader)
    trainAccuracy = accuracy_score(train_set.targets, train_preds.argmax(dim=1))
    testAccuracy = accuracy_score(test_set.targets, test_preds.argmax(dim=1))
    print(f"Accuracy on train data: {trainAccuracy}")
    print(f"Accuracy on test data: {testAccuracy}")
