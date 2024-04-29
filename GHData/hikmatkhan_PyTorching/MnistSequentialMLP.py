import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

IS_GPU_AVAILABLE = False
if (torch.cuda.is_available()):
    IS_GPU_AVAILABLE = True
else:
    IS_GPU_AVAILABLE = False

train_set = torchvision.datasets.MNIST("./data",
                                       train=True, transform=transform.Compose([transform.ToTensor()]), download=True)
test_set = torchvision.datasets.MNIST("./data",
                                      train=False, transform=transform.Compose([transform.ToTensor()]), download=True)

print("Train set info:", len(train_set))
print("Test set info:", len(test_set))

INPUT_DIM = train_set[0][0].numpy().shape
BATCH_SIZE = 1024 * 4
EPOCH = 10

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_laoder = DataLoader(test_set, batch_size=BATCH_SIZE)

MLP_Sequential = nn.Sequential(
    nn.Linear(in_features=INPUT_DIM[1] * INPUT_DIM[1], out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=10)
)
print("===Sequential Model Info===")
print(MLP_Sequential)

if IS_GPU_AVAILABLE:
    MLP_Sequential = MLP_Sequential.cuda()

loss_fn = CrossEntropyLoss()
adam = optim.Adam(MLP_Sequential.parameters(), lr=0.01)

print("===Training===")
for epoch in tqdm(range(EPOCH)):
    for i, data in enumerate(train_loader, 0):
        batch_x, batch_y = data
        batch_x = batch_x.view(-1, INPUT_DIM[0] * INPUT_DIM[1] * INPUT_DIM[2])
        if IS_GPU_AVAILABLE:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        output = MLP_Sequential(batch_x)
        loss = loss_fn(output, batch_y)
        if i % 5 == 0:
            print("Loss: ", loss)
        adam.zero_grad()
        loss.backward()
        adam.step()


print("===Evalution===")
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
classes = ('0', '`1`', '2', '3',
           '4', '5', '6', '7', '8', '9')
with torch.no_grad():
    for i, data in enumerate(test_laoder, 0):
        batch_x, batch_y = data
        batch_x = batch_x.view(-1, INPUT_DIM[0] * INPUT_DIM[1] * INPUT_DIM[2])
        if IS_GPU_AVAILABLE:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        prediction = MLP_Sequential(batch_x)
        _, predicted = torch.max(prediction, 1)
        c = (predicted == batch_y).squeeze()
        for i in range(10):
            label = batch_y[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


