import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

dataset = torchvision.datasets.flowers102.Flowers102("flowers", download=True)
transforms = [torchvision.transforms.Resize((224, 224)),
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]


class FlowerDataset(Dataset):
    def __init__(self, dataset, transforms):
        super().__init__()
        self.dataset = dataset
        self.transforms = torchvision.transforms.Compose(transforms)

    def __getitem__(self, item):
        datapoint = self.dataset[item]
        image, label = datapoint
        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.dataset)


dataset = FlowerDataset(dataset, transforms)
# dataloader = DataLoader(dataset, shuffle=True, batch_size=4)
#
# datapoint = next(iter(dataloader))
# print(datapoint[1])
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)



