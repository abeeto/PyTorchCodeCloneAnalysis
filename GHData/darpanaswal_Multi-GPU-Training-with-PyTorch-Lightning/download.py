import torchvision as torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    torchvision.datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(), download=True)
