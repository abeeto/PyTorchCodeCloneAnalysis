import os
from sys import argv

from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from Model import *

MODELS = {
    "FNN": FNN,
    "LeNet": LeNet,
    "AlexNet": AlexNet,
    "VGG16": VGG16,
    "ResNet": ResNet34
}


def main(args):
    if len(args) != 1 and args[1] == 'view':
        batch_size = 1
        device = 'cpu'
        view = True
    else:
        batch_size = 1024
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        view = False

    models = [i for i in os.listdir(os.curdir) if i.split(os.extsep)[-1] == 'pth']
    for idx, i in enumerate(models):
        print(f"{idx + 1}: {i}")
    selected = int(input())
    with open(models[selected - 1], 'rb') as f:
        model = torch.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1), inplace=True)
    ])
    test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for image, label in tqdm(test_dataloader):
            image = image.to(device)
            label = label.to(device)
            prediction = torch.argmax(model(image), dim=1)
            for idx in range(len(label)):
                class_total[label[idx]] += 1
                if label[idx] == prediction[idx]:
                    class_correct[label[idx]] += 1
            if view:
                plt.title(f"Model Prediction - {classes[prediction[0]]}\nAnswer - {classes[0]}")
                plt.imshow(torch.transpose(torch.transpose(image[0], 0, 2), 0, 1).numpy())
                plt.show()
    for idx in range(len(classes)):
        print(f"class {classes[idx]} accuracy: {class_correct[idx] / class_total[idx] * 100:.5f}%")
    print(f"total accuracy: {sum(class_correct) / sum(class_total) * 100:.5f}%")


if __name__ == '__main__':
    main(argv)
