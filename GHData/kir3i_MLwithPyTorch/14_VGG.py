# VGG는 Oxford VGG에서 만든 Convolution network
# torchvision.models.vgg를 import해서 사용 가능
# 이 코드는 VGG를 이용하여 CIFAR10을 학습한다.

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.models import vgg
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # multiprocessing을 위해 필요한 함수

    # set visdom (network 문제 추후 확인 바람)
    """
    import visdom

    vis = visdom.Visdom()
    vis.close(env='main')
    """

    # define loss tracker
    """
    def loss_tracker(loss_plot, loss_value, num):
        # num, loss_value are Tensor
        vis.line(X=num, Y=loss_value, win=loss_plot, update='append')
    """

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(777)
    if device == "cuda":
        torch.cuda.manual_seed_all(777)

    # set hyperparameters
    batch_size = 512
    learning_rate = 0.005

    # get dataset
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )  # transform 방법 정의 (normalize를 적용함.)

    cifar10_train = torchvision.datasets.CIFAR10(
        root="CIFAR10_data/", train=True, transform=trans, download=True
    )  # trainset
    cifar10_test = torchvision.datasets.CIFAR10(
        root="CIFAR10_data/", train=False, transform=trans, download=True
    )  # testset

    data_loader = torch.utils.data.DataLoader(
        dataset=cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )  # data loader
    test_loader = torch.utils.data.DataLoader(
        dataset=cifar10_test,
        batch_size=len(cifar10_test),
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )  # test data loader

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )  # 각 숫자가 어떤 object를 나타내는지 저장하는 tuple

    # image를 표시하기 위한 부분
    """
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(" ".join(f"{classes[labels[j]] for j in range(4):5s}"))
    """
    # 어떻게 네트워크를 구성할 것인가?
    cfg = [
        32,
        32,
        "M",
        64,
        64,
        128,
        128,
        128,
        "M",
        256,
        256,
        256,
        512,
        512,
        512,
        "M",
    ]  # 13+3 = vgg16

    class VGG(nn.Module):
        def __init__(self, features, num_classes=1000, init_weights=True):
            super(VGG, self).__init__()
            self.features = features
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            if init_weights:
                self._initialize_weights()

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        def _initialize_weights(self):
            for mo in self.modules():
                if isinstance(mo, nn.Conv2d):  # convolution weight 초기화
                    nn.init.kaiming_normal_(
                        mo.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if mo.bias is not None:
                        nn.init.constant_(mo.bias, 0)  # bias를 0으로 만들어준다. (VGG는 bias가 0)
                elif isinstance(mo, nn.BatchNorm2d):
                    nn.init.constant_(mo.weight, 1)
                    nn.init.constant_(mo.bias, 0)
                elif isinstance(mo, nn.Linear):
                    nn.init.normal_(mo.weight, 0, 0.01)
                    nn.init.constant_(mo.bias, 0)

    model = VGG(vgg.make_layers(cfg), 10, True).to(device)
    # vgg.make_layers는 입력한 list에 따라 convolution layer를 만들어준다.
    # VGG를 customize하느라고 VGG class를 따로 선언했다.

    # set optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_sched = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.9
    )  # 5 step마다 lr에 gamma값 곱함 (점점 작아져서 정확도 상승)

    # make plot
    """
    loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
    """

    # learning
    batches = len(data_loader)
    nb_epochs = 50

    for epoch in range(nb_epochs):
        avg_cost = 0
        for X, Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)

            # hypothesis
            hypothesis = model(X)

            # cost
            cost = F.cross_entropy(hypothesis, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            avg_cost += cost / batches
        lr_sched.step()  # lr 조정
        print(f"epoch: {epoch+1:2d}/{nb_epochs} Cost: {avg_cost:.6f}")
        # update loss_tracker
        """
        loss_tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch]))
        """
    print("Learning Finished")
    torch.save(model.state_dict(), "./model/VGG16.pth")  # 학습된 model을 저장한다!
    # new_model = CNN().to(device)
    # new_model.load_state_dict(torch.load('./model/model.pth'))  # 저장된 model을 load한다!

    # accuracy
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)

            # hypothesis
            hypothesis = model(X)
            # accuracy
            isCorrect = torch.argmax(hypothesis, dim=-1) == Y
            accuracy = torch.mean(isCorrect.float())

            print(f"Accuracy: {accuracy*100:.3f}")

"""
후기
로컬에서 돌리려니까 시간이 너무 오래걸려서 colab에서 돌려보았다.
 -> colab이 더 느려서 그냥 로컬에서 돌렸다.
약 30분 정도 걸린 것 같다.
epoch 50회: 75.460%의 정확도를 보였다.
작업관리자로 보니 CPU 사용률 10%, GPU 사용률 2%에 그치던데
사용률을 더 높일 방법은 없을까?
"""

