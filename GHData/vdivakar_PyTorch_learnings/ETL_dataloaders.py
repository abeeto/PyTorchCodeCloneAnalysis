import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

train_set = torchvision.datasets.FashionMNIST(
    root = './data/',
    train=False, #should be True though
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size = 10
)

print(len(train_set))
print(train_set.train_labels) #FashionMNIST's labels are numbered 0-9 corresponding to items
print(train_set.train_labels.bincount()) #FashionMNIST is a balanced dataset i.e. equal class samples

'''Getting Sample'''
sample = next(iter(train_set))
print("Length of sample: ", len(sample))
print("Type of sample: ", type(sample))
image, label = sample 
''' ** Sequence Unpacking ** 
    instead of writing
      image = sample[0]
      label = sample[1]
'''
print("Image shape: ", image.shape)
try:
    print("Label shape: ", label.size())
except:
    print("[X] Int type doesn't have size or shape method")

# plt.imshow(image.squeeze(), cmap='gray')
# plt.show()

my_iter = iter(train_set)
sample2 = next(my_iter)
img2, label2 = sample2
# plt.imshow(img2.squeeze(), cmap='gray')
# plt.show()

'''Getting Batch using Dataloader'''
batch = next(iter(train_loader))
print("Batch list size: ",len(batch)) # 2
print("Batch type: ", type(batch))# list
imageS, labelS = batch
print("imageS batch shape: ", imageS.shape) # [batch_size, 1, 28, 28]
print("labelS batch shape: ", labelS.shape) # [batch_size]


grid = torchvision.utils.make_grid(imageS, nrow=10)
plt.figure(figsize=(15,15))
# # plt.imshow(np.transpose(grid, (1,2,0)))
# plt.imshow(grid)
# plt.show()