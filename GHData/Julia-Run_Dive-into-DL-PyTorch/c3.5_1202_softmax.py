import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
import d2lzh_pytorch as d2l
import time

mnist_train = torchvision.datasets.FashionMNIST(root="~/Datasets/FashionMNIST", download=False, train=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root="~/Datasets/FashionMNIST", download=False, train=True,
                                               transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label, type(feature), type(label), sep='    ')


# label to number
def get_fashion_mnist_label(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankel boot']
    return [text_labels[int(i)] for i in labels]


# show
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, title in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())
        f.set_title(title)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# test show
img, labels = [], []
for i in range(10):
    img.append(mnist_train[i][0])
    labels.append(mnist_train[i][1])

l = get_fashion_mnist_label(labels)
show_fashion_mnist(img, l)

# iter
batch_size = 10
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers)

for x, y in train_iter:
    print(x.shape)
    print(get_fashion_mnist_label(y))
    break

start = time.time()
for x, y in train_iter:
    continue
print(time.time() - start)
