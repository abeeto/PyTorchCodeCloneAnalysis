# /commands.py
from pydantic import BaseModel
from torchvision import datasets

dataset = datasets.FashionMNIST("./data", download=True, train=False)


class HyperParam(BaseModel):
    model_name: str = "fashionMNIST"
    criterion: dict = {"name": "CrossEntropyLoss"}
    optimizer: dict = {"name": "Adam", "lr": 0.001}
    batch_size: int = 64
    n_epoch: int = 1
    n_labels: int = len(dataset.classes)
    transform_1D: dict = {
        "ToTensor": True,
        "Resize": (28, 28),
        "Normalize": {"mean": (0.5,), "std": (0.5,)},
    }
    transform_3D: dict = {
        "ToTensor": True,
        "Resize": (32, 32),
        "Normalize": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    }
    label_names: list = dataset.classes
    initial_hidden_size: int = 32


if __name__ == "__main__":
    print(HyperParam())
