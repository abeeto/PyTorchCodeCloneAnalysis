import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, models, transforms as Transforms

from trainer import Model
from tools import AvgMeter, AvgMeterVector

from sklearn.metrics import f1_score, accuracy_score


class MyModel(Model):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
        self.model = model

    def forward(self, x):
        x = torch.cat([x] * 3, dim=1)
        return self.model(x)

    def update_metrics(self, preds, target):
        metrics_ = self.get_metrics()
        metrics = metrics_[self.current_epoch]
        if metrics.get("Accuracy", None) == None:
            metrics["Accuracy"] = AvgMeterVector(10)
            metrics["F1_Score"] = AvgMeterVector(10)
        preds = preds.argmax(dim=1).cpu()
        target = target.cpu()
        counts = pd.Series(target).value_counts().to_dict()
        preds_onehot = F.one_hot(preds, num_classes=10)
        target_onehot = F.one_hot(target, num_classes=10)

        f1_list, acc_list = [], []
        for i in range(3):
            f1_list.append(f1_score(target_onehot[:, i], preds_onehot[:, i]))
            acc_list.append(accuracy_score(target_onehot[:, i], preds_onehot[:, i]))

        metrics["Accuracy"].update(acc_list, counts)
        metrics["F1_Score"].update(f1_list, counts)


transforms = Transforms.Compose([Transforms.ToTensor(),])

train_dataset = datasets.MNIST(
    root="C:\Moein\AI\Datasets", train=True, download=False, transform=transforms
)
valid_dataset = datasets.MNIST(
    root="C:\Moein\AI\Datasets", train=False, download=False, transform=transforms
)


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.data = [data for data in dataset]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


model = MyModel()
model.fit(
    MyDataset(train_dataset),
    MyDataset(valid_dataset),
    nn.CrossEntropyLoss(),
    5,
    512,
    file_name="mnist.pt",
)

