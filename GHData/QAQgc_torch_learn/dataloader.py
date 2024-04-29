import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

img, target = test_data[0]
# 测试数据集中第一张图片及target
print(img.shape)
print(target)

writer = SummaryWriter("logs/dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, target = data
        # print(imgs.shape)
        # print(target)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()