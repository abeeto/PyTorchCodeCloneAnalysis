from ast import Lambda
import torch
from torch.utils.data import dataset
from torchvision import  datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


training_data =  datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    target_transform=Lambda(Lambda y: torch)
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

fig = plt.figure(figsize=(8,8))
cols, rows = 3,3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label  = training_data[sample_idx]
    fig.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()

training_data = DataLoader(training_data, batch_size=64, shuffle=True)
testing_data = DataLoader(testing_data, batch_size=64, shuffle=True)

# Displaying image and label

train_features, train_labels = next(iter(training_data))
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.show()
print(f"label :{label}")
print(f"shape of image {img.shape}")

