import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root = "./dataset", train=True, transform=torchvision.transforms.ToTensor(),download=False)

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter("logs")
step = 0
for img, targets in test_loader:
    if step == 10:
        break
    writer.add_images("test_data",img,step)
    step = step+1
writer.close()

# for img, targets in test_loader:
#     print(img.shape, targets, sep=' ')
    # torch.Size([4, 3, 32, 32]) tensor([3, 9, 6, 9])
# torch.Size([4, 3, 32, 32]) img.shape:批量大小，通道数，高，宽
# tensor([3, 9, 6, 9])