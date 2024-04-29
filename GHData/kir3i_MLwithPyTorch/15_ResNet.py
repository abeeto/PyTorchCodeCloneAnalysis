# ResNet은 최대 152개의 Layer를 이용하여 학습하는 model이다.
# VGG보다 발전된 model로 더 높은 정확도를 보인다.

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    """ 왜인지 동작하지 않는 visdom(시각화)
    import visdom

    vis = visdom.Visdom()
    vis.close(env='main')

    def value_tracker(value_plot, value, num):
        vis.line(X=num, Y=value, win=value_plot, update='append')

    """

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(777)
    if device == "cuda":
        torch.cuda.manual_seed_all(777)

    # set hyperparameters
    batch_size = 128
    learning_rate = 0.1

    ## transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # How to Calculate mean and std in Normalize
    trans = transforms.Compose([transforms.ToTensor()])

    cifar10_train = torchvision.datasets.CIFAR10(
        root="CIFAR10_data/", train=True, transform=trans, download=True
    )
    print(cifar10_train.data.shape)

    cifar10_train_mean = cifar10_train.data.mean(axis=(0, 1, 2))
    cifar10_train_std = cifar10_train.data.std(axis=(0, 1, 2))
    print(cifar10_train_mean)
    print(cifar10_train_std)

    cifar10_train_mean /= 255
    cifar10_train_std /= 255
    print(cifar10_train_mean)
    print(cifar10_train_std)

    # Normalize 적용하여 dataset load
    trans_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # img를 랜덤하게 잘라 32*32로 출력
            transforms.ToTensor(),
            transforms.Normalize(cifar10_train_mean, cifar10_train_std),
        ]
    )

    trans_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_train_mean, cifar10_train_std),
        ]
    )

    cifar10_train = torchvision.datasets.CIFAR10(
        root="CIFAR10_data/", train=True, transform=trans_train, download=True
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root="CIFAR10_data/", train=False, transform=trans_test, download=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=cifar10_test,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    """
    classes = ('plane', 'car', 'bird', 'cat', deer',
        'dog', 'frog', 'horse', 'ship', 'truck')
    """

    # set model
    conv1x1 = resnet.conv1x1
    Bottleneck = resnet.Bottleneck
    BasicBlock = resnet.BasicBlock

    # 연습 차원에서 ResNet 클래스 다시 작성
    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
            super().__init__()
            self.inplanes = 16
            self.conv1 = nn.Conv2d(
                3, 16, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
            self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128 * block.expansion, num_classes)

            for mo in self.modules():
                if isinstance(mo, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        mo.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(mo, nn.BatchNorm2d):
                    nn.init.constant_(mo.weight, 1)
                    nn.init.constant_(mo.bias, 0)
            if zero_init_residual:
                for mo in self.modules():
                    if isinstance(mo, Bottleneck):
                        nn.init.constant_(mo.bn3.weight, 0)
                    elif isinstance(mo, BasicBlock):
                        nn.init.constant_(mo.bn2.weight, 1)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            # 1, 3, 32, 32
            x = self.conv1(x)
            # 1, 16, 32, 32
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            # 1, 128, 32, 32
            x = self.layer2(x)
            # 1, 256, 32, 32
            x = self.layer3(x)
            # 1, 512, 16, 16
            x = self.layer4(x)
            # 1, 1024, 8, 8
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    model = ResNet(Bottleneck, [3, 4, 6, 3], 10, True).to(device)
    # print(model)

    # set optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    """ make plot
    loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
    acc_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Accuracy', legend=['Acc'], showlegend=True))
    """

    # define accuracy check function
    def acc_check(model, test_set, epoch, save=True):
        correct = 0
        with torch.no_grad():
            # testset을 통째로 넣으면 그래픽카드 전용 램이 용량을 초과해서
            # mini batch로 쪼개서 test 해야 한다.
            for X, Y in test_set:
                X = X.to(device)
                Y = Y.to(device)

                hypothesis = model(X)
                isCorrect = torch.argmax(hypothesis, dim=-1) == Y
                correct += torch.sum(isCorrect.float()).item()

            accuracy = 100.0 * correct / len(cifar10_test)
            print(f"epoch: {epoch+1:3d}/{nb_epochs} Accuracy: {accuracy:.3f}")
            if save:
                torch.save(
                    model.state_dict(),
                    f"./model/ResNet/resnet50_epoch_{epoch+1}_acc_{int(accuracy)}.pth",
                )
            return accuracy

    old_model = "resnet50_epoch_15_acc_0.pth"  # 저장된 모델 불러오기
    if old_model != "":
        model.load_state_dict(
            torch.load("./model/ResNet/" + old_model)
        )  # 저장된 model을 load한다!
        print(f"model loaded ---> {old_model}")

    # learning
    batches = len(train_loader)
    nb_epochs = 50

    # acc = acc_check(model, test_loader, 0, save=False)  # test

    for epoch in range(16, nb_epochs):
        avg_cost = 0
        for X, Y in train_loader:
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
        print(f"epoch: {epoch+1}/{nb_epochs} cost: {avg_cost:.6f}")
        acc = acc_check(model, test_loader, epoch, save=True)
        # value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch+1]))
        lr_sched.step()
    print("Learning Finished")

"""
후기
VGG보다 학습속도가 빠르다고 하던데 딱히 그런진 모르겠다.
epoch 50회로 학습 시 최대 88.010%의 정확도를 보였다.
"""

