import torch
import torch.nn as nn
from torchvision import transforms,datasets
import torch.nn.functional as F
import numpy
from torch.autograd import Variable

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(100),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# dataset = datasets.ImageFolder(root=r'C:\Users\Lavector\Desktop\kaggle\train')
# # dataset = datasets.ImageFolder(root=r'C:\Users\Lavector\Desktop\项目\分类')
#
# # cat文件夹的图片对应label 0，dog对应1
# print(dataset.class_to_idx)

train_dataset = datasets.ImageFolder(root=r'/Users/dingweiqi/Desktop/datas/train/', transform=data_transform)
test_dataset = datasets.ImageFolder(root=r'/Users/dingweiqi/Desktop/datas/test/', transform=data_transform)
# test_dataset = datasets.ImageFolder(root=r'C:\Users\Lavector\Desktop\项目\分类', transform=data_transform)
# test_dataset = datasets.ImageFolder(root=r'C:\Users\Lavector\Desktop\项目\分类test', transform=data_transform)
# print(train_dataset)
# train_dataset=torch.utils.data.TensorDataset(train_dataset,"cat")
# train_loader = torch.Tensor(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=100, shuffle=True,num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=8,shuffle=False,num_workers=1)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(1,32,11,1,5),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        self.con2= nn.Sequential(
            nn.Conv2d(32,64, 11, 1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # self.con3 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        self.fc = nn.Sequential(
            nn.Linear(64*25*25,200),
            nn.ReLU(),
            nn.Linear(200,20),
            nn.ReLU(),
            nn.Linear(20,2)
        )
        self.opt = torch.optim.Adam(self.parameters())
        self.los = torch.nn.CrossEntropyLoss()


    def forward(self, input):
        out=self.con1(input)
        out=self.con2(out)
        # out=self.con3(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out


    def train_modle(self, x, y):
        out=self.forward(x)
        loss=self.los(out,y)
        print("loss:",loss.item())
        print("loss:",loss)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def test_modle(self, x):
        return self.forward(x)


net=MyNet()
# net=MyNet().cuda()
# print(type(train_loader))
if __name__ == '__main__':
    for epoch in range(10):
        print("===epoc===%d"%epoch)
             # running_loss = 0.0
        for i,(data,y) in enumerate(train_loader):
            # train_loader = Variable(data)
            # y = Variable(y)

            # print(i,data,y)
            print(y)
            net.train_modle(data,y)
            # net.train_modle(data.cuda(),y.cuda())

    net.eval()
    # yy=net.test_modle(data)
    # print(yy)
    testacc = 0
    for datas in test_loader:
        images, labels = datas
        # print(net.test_modle(images))
        torch.no_grad()
        print(labels)
        outputs=net.test_modle(images)
        predict = F.softmax(outputs)
        # predict = F.softmax(net.test_modle(images.cuda()))
        # print(predict)
        # print(torch.max(predict,1))#返回每行中最大的数，并返回其索引值
        # print("%.2f"%torch.max(predict).item())
        # print(torch.max(predict,1)[1].data.numpy() )
        # print(torch.max(predict,1)[0].data.numpy() )
        # print(type(torch.max(predict,1)[0].data.numpy().tolist() ))
        # print(zip(torch.max(predict,1)[0].data.numpy().tolist(),torch.max(predict,1)[1].data.numpy().tolist()))

        mm, prediction = torch.max(outputs.data, 1)
        print(outputs)
        print(mm)
        print(prediction)
    #     print(labels)
    #     print(labels.data)
    #
        testacc += torch.sum(prediction == labels.data)
    #     print(testacc)
    print(testacc)

        # _, predicted = torch.max(net.test_modle(images).data, 1)
        # num_correct = (predicted == labels).sum()
        # testacc += num_correct.item()
        # # _, predicted = torch.max(net.test_modle(images.cuda()).item(), 1)
        # print(predicted,testacc)
