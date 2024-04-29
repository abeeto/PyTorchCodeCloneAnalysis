from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

batch_size = 64

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

mnist = datasets.MNIST("./data", transform=img_transform, download=False)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=4)
