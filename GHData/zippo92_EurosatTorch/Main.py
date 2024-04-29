from torch.utils.data import DataLoader
from EurosatDataset import EurosatDataset
from torchvision import transforms

if __name__ == '__main__':
    data_transform = transforms.Compose([
        # numpy ndarray to PIL because resize needs a PIL image in input
        transforms.ToPILImage(),
        transforms.Resize(128),
        # # numpy image: H x W x C -> torch image: C X H X W -> Tensor
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    trainingDataset = EurosatDataset("Dataset/eurosat_prova/train", data_transform)

    train_loader = DataLoader(trainingDataset, batch_size=10, shuffle=True, num_workers=1)
