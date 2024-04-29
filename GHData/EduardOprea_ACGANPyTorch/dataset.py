import torch
from torchvision.transforms import transforms
from torchvision import datasets

def load_data(path, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    train_dataset = datasets.ImageFolder(root=path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader