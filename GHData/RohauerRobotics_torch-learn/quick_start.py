import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# download datasets from open data
# data which is already downloaded doesn't need to be downloaded a second time
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = False,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = False,
    transform = ToTensor()
)

# define batchsize
# batchsize for the DataLoader represnts the number of samples of features or labels
# processed at any given time

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# displays the shape of the DataLoader iterable created
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# define device variable name based on the availability of gpu
# processing
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usinging {device} device")

# declare class of neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        # super says that NeuralNetwork will inherit nn.Module's
        # functions and class variables
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# moves the buffer space to the declared processing device (ideally gpu)
model = NeuralNetwork().to(device)
print(model)

# define the loss function for the NeuralNetwork
loss_fn = nn.CrossEntropyLoss()

# define the optimization function for minimizing loss
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# define training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # after batch is processed show loss number
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:7f} [{current:5d}/{size:5d}]")

# define testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error \n Accuracy: {correct*100:>0.1f} % , Avg Loss: {test_loss:>8f}\n")

# iterate through 5 epochs to improve accuracy
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done")

torch.save(model.state_dict(),"model.pth")
print("Saved torch model to model.pth")

# make new empty model to load saved model
model = NeuralNetwork()
# load saved model into neural network
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted}, Actual: {actual}")
