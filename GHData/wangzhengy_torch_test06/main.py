import os

import torch

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
if __name__ == '__main__':
    DEVICE = torch.device('cuda:0')

    train_path = 'data/train/'
    labels = os.listdir(train_path)

    class Data(Dataset):
        def __init__(self, root_dir, label_dir,transform=None):
            self.transform = transform
            self.root_path = root_dir
            self.labels = os.listdir(self.root_path)
            # self.img_path[self.labels] = os.listdir(os.path.join(self.train_path,self.labels))
            self.img_names = []
            for label in self.labels:
                self.img_names += os.listdir(os.path.join(self.root_path, label))
            # print(self.img_names)

        def __getitem__(self, idx):

            img_name = self.img_names[idx]
            label = img_name.split('_')[0]
            img_item_path = os.path.join(self.root_path, label, img_name)
            img = Image.open(img_item_path)

            transf = transforms.ToTensor()
            img = transf(img)

            targets = labels.index(label)
            targets = torch.tensor(targets)
            # targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # targets[labels.index(label)] += 1
            # targets = torch.Tensor(targets)

            # targets = eval(img_name.split('.')[0].split('_')[2])
            # targets = torch.tensor(targets)

            return img, targets

        def __len__(self):
            return len(self.img_names)

    # train_data = torchvision.datasets.CIFAR10(root='../data',train=True,transform=torchvision.transforms.ToTensor(),
    #                                           download=True)
    # test_data = torchvision.datasets.CIFAR10(root='../data',train=False,transform=torchvision.transforms.ToTensor(),
    #                                          download=True)

    train_data = Data('data/train/','')
    test_data = Data('data/train','')


    train_data_size = len(train_data)
    test_data_size = len(test_data)

    print('训练数据集长度为{}'.format(train_data_size))
    print('测试数据集长度为{}'.format(test_data_size))

    train_dataloader = DataLoader(train_data,batch_size=16,shuffle=True, pin_memory=True)

    test_dataloader = DataLoader(test_data,batch_size=1,shuffle=False, pin_memory=True)

    class Model(nn.Module):
        def __init__(self):
            super(Model,self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3,32,5,1,2,bias=False),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
                nn.Conv2d(32,32,5,1,2,bias=False),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
                nn.Conv2d(32,64,5,1,2,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64*4*4*4,14),
            )


        def forward(self,x):
            x = self.model(x)
            return x



    trainModel = Model()
    # trainModel = torch.load('train_model', map_location=torch.device('cpu'))
    trainModel = trainModel.cuda()

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()

    learning_rate = 0.05 #0.04-
    optim = torch.optim.SGD(trainModel.parameters(),lr=1e-3, weight_decay=5e-4)

    total_train_step = 0

    total_test_step = 0

    epoch = 1500 #2500

    trainModel.train()

    for i in range(epoch):
        print('-----第',i+1,'轮训练开始-----')
        for data in train_dataloader:
            imgs,targets = data
            # print(imgs)
            # print(targets)
            imgs = imgs.cuda()
            targets = targets.cuda()
            # print('targets',targets)
            outputs = trainModel(imgs)
            # print('outputs',outputs)
            loss = loss_fn(outputs,targets)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_train_step += 1
            # if total_train_step % 100 == 0:
            #     print('训练次数为:',total_train_step,' loss:',loss.item())
        trainModel.eval()
        total_test_loss = 0
        total_accurary = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs,targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                outputs = trainModel(imgs)
                loss = loss_fn(outputs,targets)
                total_test_loss += loss.item()
                # print(targets)
                accurary = (outputs.argmax(1) == targets.item()).sum()

                total_accurary = total_accurary + accurary

                # if total_accurary != 0:
                #     print(outputs)
                #     print(outputs.argmax(1))
                accurary = 0
            print('整体测试集上的Loss:',total_test_loss)
            print(total_accurary)
            print('整体测试集上的acc:',total_accurary/test_data_size)
        if i % 100 == 0:
            torch.save(trainModel,'train_model_1')
    print('模型已保存')
