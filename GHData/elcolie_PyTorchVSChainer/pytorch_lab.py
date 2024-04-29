import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
import multiprocessing as mp

NUM_PROCS = mp.cpu_count()


class SimpleGRUClassifier(nn.Module):
    """GRU without lightning."""

    def __init__(
        self,
        n_classes: int = 2,
        input_size: int = 1000,
        hidden_layer_size: int = 500,
        n_layers: int = 1,
        hidden_size: int = 1,
        batch_size: int = 1,
    ):
        """Instantiate the instance."""
        super().__init__()
        # self.n_classes = len(label_encoder.classes_)
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(
            input_size, hidden_layer_size, n_layers, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc = nn.Linear(hidden_layer_size, self.n_classes)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computing."""
        hidden = self._init_hidden(self.hidden_layer_size)
        output, hidden = self.gru(x, hidden.view(self.n_layers, -1))
        fc_output = self.relu(self.fc1(output))
        fc_output = self.fc(fc_output)
        return fc_output

    def _init_hidden(self, batch_size):
        """Return initial condition of hidden state tensor."""
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden.clone().detach().to(device)


def train(
    model: SimpleGRUClassifier,
    device: str,
    criterion: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    optimizer: torch.optim.Adam,
    epoch: int,
):
    """Train the neural network without Lightning."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(
    model: SimpleGRUClassifier,
    device: str,
    criterion: nn.CrossEntropyLoss,
    test_loader: DataLoader,
):
    """Test the neural network without lightning."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )


class MushroomDataSet(Dataset):
    """Dataset."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ):
        """Initialize the instance."""
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __getitem__(self, index):
        """Get an item from given index."""
        return torch.Tensor(
            self.X[index]
        ), int(self.Y[index])

    def __len__(self):
        """Return length of dataset."""
        return self.len


def main() -> None:
    """Run the code."""
    mushroomsfile = 'mushrooms.csv'
    data_array = np.genfromtxt(
        mushroomsfile, delimiter=',', dtype=str, skip_header=1)
    for col in range(data_array.shape[1]):
        data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]

    X = data_array[:, 1:].astype(np.float32)
    Y = data_array[:, 0].astype(np.int32)[:, None]
    df = pd.DataFrame(X)
    df['label'] = Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=112)

    train_dataset = MushroomDataSet(
        X_train,
        y_train
    )
    test_dataset = MushroomDataSet(
        X_test,
        y_test
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=NUM_PROCS,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=NUM_PROCS,
    )
    model = SimpleGRUClassifier(n_classes=2, input_size=22)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, 1 + 1):
        train(model, device, criterion, train_loader, optimizer, epoch)
        test(model, device, criterion, test_loader)
        scheduler.step()


if __name__ == "__main__":
    main()
