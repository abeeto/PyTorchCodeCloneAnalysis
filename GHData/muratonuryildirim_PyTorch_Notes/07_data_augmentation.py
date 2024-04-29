import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
from torchvision.utils import save_image

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1
num_epochs = 10

# load dataset
class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label


my_transforms = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((256,256)),
                                    transforms.RandomCrop((224,224)),
                                    transforms.ColorJitter(brightness=0.5),
                                    transforms.RandomRotation(degrees=45),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
                                    ])

dataset = CatsAndDogsDataset(csv_file='dataset/cats_dogs/cats_dogs.csv', root_dir='dataset/cats_dogs', transform=my_transforms)

train_set, test_set = torch.utils.data.random_split(dataset, [8, 2])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

#loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
print(f'Training started with {num_epochs} epoch(s)')

for epoch in range(num_epochs):

    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # evaluation state on, train state off

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    model.train()  # evaluation state off, train state on


# accuracy check
print('Checking accuracy on training dataset...')
check_accuracy(train_loader, model)
print('Checking accuracy on test dataset...')
check_accuracy(test_loader, model)
