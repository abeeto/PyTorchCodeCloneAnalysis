import torch
from torchvision import datasets, transforms

DATAPATHS = {
    'EMNIST': '../dataEMNIST',
    'MNIST': '../dataMNIST'

}


def load_emnist(batch_size):
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1722,), (0.3310,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(DATAPATHS['EMNIST'], split="byclass", train=True, download=True, transform=transformations),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    labels_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(DATAPATHS['EMNIST'], split="byclass", train=False, download=False, transform=transformations),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    dataset_test_len = len(labels_loader.dataset)
    dataset_train_len = len(train_loader.dataset)
    print("Длина обучающего датасета {}\n Длина трениро"
          "вочного датасета\n".format(dataset_train_len, dataset_test_len))
    return train_loader, labels_loader


def load_mnist(batch_size):
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATAPATHS['MNIST'], train=True, download=True, transform=transformations),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    labels_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATAPATHS['MNIST'], train=False, download=False, transform=transformations),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, labels_loader


LOADERS = {
    'EMNIST': load_emnist,
    'MNIST': load_mnist,
}

OPTIONS = {
    'EMNIST':
        {
            'need_resize': True
        },
    'MNIST':
        {
            'need_resize': True
        },
}


class DataLoader:
    def __init__(self, dataset_name, batch_size):
        """
        :param batch_size: batch size used in loader

        :param dataset_name: name of loaded dataset


        """
        actual_name = dataset_name
        if dataset_name not in LOADERS:
            actual_name = "MNIST"

        self.dataset_name = actual_name
        self.batch_size = batch_size
        self.need_resize = OPTIONS[dataset_name]['need_resize']

    @property
    def dataset(self):
        """
        Returns dataset loaders
        :return: (train_loader, test_loader)
        """
        return LOADERS[self.dataset_name](self.batch_size)
