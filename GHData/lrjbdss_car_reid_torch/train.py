from model import EmbeddingNet
from resnet import resnet18
from dataset_loader import ImageDataset, train_dataset
from transform import Transform, calculate_mean_and_std
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import random
import logging

logging.basicConfig(level=10,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    filename='log/resnet18_train.log')
pretrained_model = './weights/59_custom.pth'
train_file = 'new_train_list.txt'
# test_file = '3200_test_list.txt'
img_path = '/data/VehicleID_V1.0/image'
# img_path = '/media/lx/新加卷/datasets/VehicleID/image'
width = 120
height = 120
# img_mean = [0.485, 0.456, 0.406]
# img_std = [0.229, 0.224, 0.225]
img_mean = [0.3464, 0.3639, 0.3659]
img_std = [0.2262, 0.2269, 0.2279]
epoch = 100
alpha = 0.4
batch_size = 64
num_workers = 16
# accumulation_steps = 2  # 8个batch更新一次，实际的batch为 batch_size*accumulate_grad_batches

dataset = train_dataset(train_file, img_path)
random.shuffle(dataset)
trainset = dataset[:76800]
testset = dataset[76800:]
# testset = train_dataset(test_file, img_path)
trans = Transform(width, height, img_mean, img_std)
trainLoader = DataLoader(ImageDataset(trainset, transform=trans),
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers,
                         pin_memory=True)
# mean, std = calculate_mean_and_std(trainLoader, len(trainset))
# print('mean and std:', mean, std)

testLoader = DataLoader(ImageDataset(testset, transform=trans),
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True)


# net = EmbeddingNet()
net = resnet18(pretrained=False)
net = nn.DataParallel(net).cuda()
net.load_state_dict(torch.load(pretrained_model))
net.train()
for param in net.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70])

pretrain_sign = True
for e in range(epoch):

    running_loss = 0.
    for i, data in enumerate(trainLoader):
        imgs, colors, models = [p.cuda() for p in data]
        colors_pred, models_pred = net(imgs)

        loss = alpha * criterion(colors_pred, colors) + (1 - alpha) * criterion(models_pred, models)
        running_loss += loss.data
        # loss /= accumulation_steps
        loss.backward()

        # if (i+1)%accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

        # print('第%d个epoch,第%d个batch' % (e, i))

        if i % 200 == 199:
            logging.info('[%d, %5d] loss: %.3f' %
                         (e + 1, i + 1, running_loss / 200))
            running_loss = 0.
    scheduler.step(e)
    if e % 10 == 9:
        colors_correct, models_correct = 0, 0
        for j, data in enumerate(testLoader):
            imgs, colors, models = [p.cuda() for p in data]
            colors_pred, models_pred = net(imgs)
            colors_pred = colors_pred.argmax(-1)
            models_pred = models_pred.argmax(-1)
            colors_correct += (colors_pred == colors).sum()
            models_correct += (models_pred == models).sum()
        logging.info('color正确率：%3f, model正确率：%3f' % (colors_correct.float()/3200, models_correct.float()/3200))

        savePath = './weights/%d_custom.pth' % e
        torch.save(net.state_dict(), savePath)

    # if e > 10 and pretrain_sign:
    #     for param in net.parameters():
    #         param.requires_grad = True
    #     pretrain_sign = False

logging.info('Finished Training')

