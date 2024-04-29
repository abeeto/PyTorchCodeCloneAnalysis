import torch
import torchvision
from torchvision import transforms, datasets

# collect data
train = datasets.MNIST(
    '', 
    train = True, 
    download = True, 
    transform = transforms.Compose([ transforms.ToTensor() ])
)

test = datasets.MNIST(
    '', 
    train = False, 
    download = True, 
    transform = transforms.Compose([ transforms.ToTensor() ])
)

# prepare data sets for torch
# batch sizes are set to allow operation in RAM capable sizes of data
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

for data in trainset:
    print(data)
    break
