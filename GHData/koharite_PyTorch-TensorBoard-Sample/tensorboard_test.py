'''
PyTorchでTensor Boardを使うテストをしてみる。
Reference:
    https://qiita.com/nj_ryoo0/items/f3aac1c0e92b3295c101
    https://pytorch.org/docs/stable/tensorboard.html

Author:
    Takehiro Matsuda

Data:
    5th, Dec., 2019

'''

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# numpyで作ったデータを表示してみる
def test1():
    x = np.random.randn(100)
    # xの累積和
    y = x.cumsum()

    # log_dirでlogのディレクトリを指定
    writer = SummaryWriter(log_dir = "test1_logs")

    # xとyの値を記録していく
    for i in range(100):
        writer.add_scalar("x", x[i], i)
        writer.add_scalar("y", y[i], i)
    
    # writerを閉じる
    writer.close()

# x,yが複数の系列がある場合のサンプル
def test2():
    # ログをとる対象を増やしてみる
    x1 = np.random.randn(100)
    y1 = x1.cumsum() 

    x2 = np.random.randn(100)
    y2 = x2.cumsum() 

    writer = SummaryWriter(log_dir="test2_logs")

    # tagの書き方に注目！
    for i in range(100):
        writer.add_scalar("X/x1", x1[i], i)
        writer.add_scalar("Y/y1", y1[i], i)
        writer.add_scalar("X/x2", x2[i], i)
        writer.add_scalar("Y/y2", y2[i], i)
    writer.close()

# PyTorch Officialのmnistを使った例
def test3():
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir="test3_logs")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)

    # dummy data for visualize
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


    writer.close()




if __name__ == '__main__':
    # test1を実行する
    # プログラム実行後に、以下のコマンドでTensorBoardを実行し、
    # ウェブブラウザでlocalhost:6006でアクセスし結果を確認する
    # tensorboard --logdir="test1_logs"
    #test1()

    # プログラム実行後に、以下のコマンドでTensorBoardを実行し、
    # ウェブブラウザでlocalhost:6006でアクセスし結果を確認する
    # tensorboard --logdir="test2_logs"
    #test2()

    # プログラム実行後に、以下のコマンドでTensorBoardを実行し、
    # ウェブブラウザでlocalhost:6006でアクセスし結果を確認する
    # tensorboard --logdir="test3_logs"
    test3()

