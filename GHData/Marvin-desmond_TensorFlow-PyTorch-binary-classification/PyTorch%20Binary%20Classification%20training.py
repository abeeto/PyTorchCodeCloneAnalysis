# Let us import our libraries
import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Loading the cifar10 dataset
trainset = torchvision.datasets.CIFAR10(
    root="../../datasets", train=True,
    download=False, transform=transform
)

print(f"There are {len(trainset)} images and labels in our dataset")

# Plane and bird are mapped at indeces 0 and 2 respectively
# Let us create a mapper
mapper = {
    0 : "plane",
    2 : "bird"
}

os.system("cls")

# filtering data for plane and birds
train_images = np.array([np.asarray(image) for image, label in trainset if label in list(mapper.keys())])
train_labels = np.array([label for image, label in trainset if label in list(mapper.keys())])
print(f"{len(train_images)} images of planes and birds")
print(f"{len(train_labels)} labels of planes and birds")

# change labels of bird from 2 to 1
train_labels = np.where(train_labels==2, 1, train_labels)
# a new mapper
data_map = {0 : "plane",1 : "bird"}

tensor_images = torch.Tensor(train_images)
tensor_labels = torch.Tensor(train_labels)

# Visualizing a few images of planes and birds
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(tensor_images[i].permute(1, 2, 0))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.title(data_map[train_labels[i]])
plt.show()

# create a single batch from an image
single_batch = tensor_images[0].unsqueeze(0)
print(single_batch.size())

# Create a convolution layer 
import torch.nn as nn
import torch.nn.functional as F

# convolution and pooling (downsampling) the single image batch

# Input image 3 channels, 16 filters of kernel 3 * 3
conv_layer = nn.Conv2d(3, 16, 3)
# Maxpooling of stride 2 * 2
pool_layer = nn.MaxPool2d(2)

output_conv = conv_layer(single_batch.float())
output_pool = pool_layer(single_batch.float())

    
for index, i in enumerate([
    [single_batch[0], "Normal", "viridis"],
    [output_conv[0, 0].detach(), "Convoluted", "gray"],
    [output_pool[0, 0].detach(), "Downsampled", "gray"]
    ]):
    if i[1] == "Normal":
        i[0] = i[0].permute(1, 2, 0)
    plt.subplot(1, 3, index+1)
    plt.imshow(i[0], cmap=i[2])
    plt.title(i[1])
plt.show()


def show_image(x):
    plt.imshow(x)
    plt.colorbar()
    plt.show()

print(tensor_images.shape, tensor_labels.shape)
tensor_train_dataset = torch.utils.data.TensorDataset(tensor_images, tensor_labels)
train_loader = torch.utils.data.DataLoader(
    tensor_train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=2
    )

# Defining our model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # Using the formula 
        # ((n + 2p - f) / s) + 1
        # and passing the image through conv, pool, then conv, pool,
        # the image dimensions to the Linear layer is an 8 * 8
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


# tensor_labels = tensor_labels.type(torch.LongTensor)


# Let us train our model using 10 epochs
net = Net()
# Loss is CrossEntropy
criterion = nn.BCEWithLogitsLoss()
# Optimizer is Stochastic Gradient Descent with Learning rate 1e-2
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=1e-2)

for epoch in range(50):
    running_loss = 0.0
    
    for data in train_loader:
        inputs, labels = data
        labels = labels.unsqueeze(1)
        # labels = labels.unsqueeze(1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # print statistics
    print("Epoch : {}, Loss: {:.4f}".format(epoch + 1, running_loss/len(train_loader)))

print("Finished training!") 
# Saving only model's state dict
PATH = './cifar_two.pth'
torch.save(net.state_dict(), PATH)

# Saving both the parameters and the structure of model to a file
torch.save(net, "./cifar_two_net.pth")
