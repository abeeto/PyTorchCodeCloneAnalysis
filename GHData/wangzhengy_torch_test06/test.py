import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

train_path = 'data/train/'
labels = os.listdir(train_path)


class Data(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
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

        targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        targets[labels.index(label)] += 1
        targets = torch.Tensor(targets)

        # targets = eval(img_name.split('.')[0].split('_')[2])
        # targets = torch.tensor(targets)

        return img, targets

    def __len__(self):
        return len(self.img_names)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2, bias=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 64),
            nn.Linear(64, 14)
        )

    def forward(self, x):
        x = self.model(x)
        return x



test_data = Data('data/valid','')
test_dataloader = DataLoader(test_data, batch_size=1)
test_data_size = len(test_data)

if __name__ == '__main__':

    #
    # model = torch.load('train_model_1', map_location=torch.device('cpu'))
    # # trainModel.eval()
    # image_path = "data/eval/9100_010_0.png"
    # image = Image.open(image_path)
    #
    # transform = transforms.ToTensor()
    # image = transform(image)
    #
    # model = torch.load('train_model_0', map_location=torch.device('cpu'))
    # # print(model)
    # image = torch.reshape(image, (1, 3, 64, 64))
    #
    # # model.eval()  # 将模型转换为测试模型
    # with torch.no_grad():  # 可以节约我们的内存，提高性能
    #     #   output = model(image.cuda())            # gpu 上训练的模型也要在gpu上加载
    #     output = model(image)
    # print(output)
    # print(output.argmax(1))
    # print(labels[output.argmax(1).item()])
    model = torch.load('train_model_0', map_location=torch.device('cpu'))
    model.eval()
    total_test_loss = 0
    total_accurary = 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            # imgs = imgs.cuda()
            # targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # print(targets)
            accurary = (outputs.argmax(1) == targets.argmax(1)).sum()
            # print(outputs.argmax(1),targets.argmax(1))
            total_accurary = total_accurary + accurary
            # if total_accurary != 0:
            #     print(outputs)
            #     print(outputs.argmax(1))
            accurary = 0
        print('整体测试集上的Loss:',total_test_loss)
        print(total_accurary)
        print('整体测试集上的acc:',total_accurary/test_data_size)

# print('模型已保存')
