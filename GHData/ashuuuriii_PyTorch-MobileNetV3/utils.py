import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


class EMA:
    def __init__(self, model: nn.Module, decay_rate: float):
        self.model = model
        self.decay_rate = decay_rate
        self.shadow = {}
        self.restore_dict = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (
                    1.0 - self.decay_rate
                ) * param.data + self.decay_rate * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.restore_dict
            param.data = self.restore_dict[name]

        self.restore_dict = {}

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.restore_dict[name] = param.data.clone()
                param.data = self.shadow[name]


def calc_mean_sd(dataloader: DataLoader):
    ch_sum, ch_sum_sqrt, n_batches = 0, 0, 0
    for data, _ in dataloader:
        ch_sum += torch.mean(data, dim=[0, 2, 3])
        ch_sum_sqrt += torch.mean(torch.square(data), dim=[0, 2, 3])
        n_batches += 1

    mean = ch_sum / n_batches
    sd = (ch_sum_sqrt / n_batches - mean**2) ** 0.5
    return mean, sd


if __name__ == "__main__":
    torch.manual_seed(42)
    dataset = datasets.CIFAR100("data", train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=128)
    mean, sd = calc_mean_sd(dataloader)
    print(f"mean: {mean}, sd: {sd}")
