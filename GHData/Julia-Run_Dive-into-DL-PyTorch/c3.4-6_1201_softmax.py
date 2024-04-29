import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())
# download=True。如果前面的root文件位置没有该文件，则下载，否则不下载，已有时，可改成False
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
question = mnist_train[1]
print(question[0].shape, type(question), type(question[1]))
print(feature.shape)
print(label, type(label))  # int，不应该是tensor吗


# 将标签转换成文本
def get_fashion_mnist_labels(labels):  # labels=[0,1,2,3,4,5,6,7,8,9]
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankel boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):  # images---list（features）， labels--list
    # 下面定义一个可以在一行里画出多张图像和对应标签的函数。
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # 1Xlen()个子图
    # plt.subplots() is a function that returns a tuple containing a figure and axes object(s)
    # fig, ax = plt.subplots()
    # fig = plt.figure()    ax = fig.add_subplot(111)
    ############test
    # import matplotlib.pyplot as plt
    # tem,fig=plt,subplot(1,4)
    # print(type(tem),type(fig))
    # print(fig)
    for f, img, lbl in zip(figs, images, labels):  # zip(*zip()) 可迭代元素打包
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)  # label
        f.axes.get_xaxis().set_visible(False)  # 不显示坐标轴的刻度
        f.axes.get_yaxis().set_visible(False)
    plt.show()


x, y = [], []
for i in range(10):
    x.append(mnist_train[i][0])  # 第i个元组的首个元素，feature---tensor，x是list
    y.append(mnist_train[i][1])  # 第i个元组的第二个元素，label---int，y是list
# i = [k for k in range(10)]
# x = mnist_train[i, 0]
# y = mnist_train[i, 1]
show_fashion_mnist(x, get_fashion_mnist_labels(y))

# iter
batch_size = 5
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

for x, y in train_iter:
    print(x.shape)
    print(y)
    break

start = time.time()
for x, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
