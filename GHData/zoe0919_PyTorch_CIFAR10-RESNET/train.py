import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import time
from tensorboardX import SummaryWriter

#   导入数据集生成函数
from custom_dataset.dataset import MyDataSet
from custom_dataset.utils import read_split_data



'''
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''
#   Cifar-10的标签：('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
Debug = True

#   权重文件，等于''时不使用预训练权重
model_path = ''


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
#   图片根目录
root = r"C:\Users\baoti\Desktop\PyTorch_CIFAR10-RESNET\images\clean_train"
#   Debug数据集目录
debug_root = r"C:\Users\baoti\Desktop\PyTorch_CIFAR10-RESNET\images\debug_images"

#   存储训练集的所有图片路径
train_images_path = []
#   存储训练集图片对应索引信息
train_images_label = []
#   存储验证集的所有图片路径
val_images_path = []
#   存储验证集图片对应索引信息
val_images_label = []



# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

#   超参数设置
#   遍历数据集次数
EPOCH = 200
#   批处理尺寸(batch_size)
BATCH_SIZE = 128
VAILD_BATCH_SIZE = 100
#   学习率
LR = 0.1
#   学习率里程碑
Milestones = [135, 185]
#   使用的内核数
nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0])
if Debug:
    print('Using {} dataloader workers'.format(nw))
    #   如果需要Debug，不要开多线程
    nw = 0

#   训练数据集预处理
#   周围填充4，随机居中裁剪；50%几率反转；转为Tensor；RGB三层，归一化均值和方差
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#   验证集归一化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#   模型定义-ResNet
net = ResNet18().to(device)
#   加载权重
if model_path != '':
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = net.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

#   定义损失函数和优化方式
#   定义损失函数：交叉熵，多用于多分类问题
criterion = nn.CrossEntropyLoss()
#   优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#   LR修改没有生效
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=Milestones, gamma = 0.1)

writer=SummaryWriter("./logs")

# 训练
if __name__ == "__main__":
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    #   训练集
    train_set = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=transform_train)

    #   生成一个个batch进行批训练，组成batch的时候顺序打乱取
    #   打包成batch
    #   collate_fn:
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = nw,
    #                                            collate_fn=train_set.collate_fn)

    test_set = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=VAILD_BATCH_SIZE, shuffle=False, num_workers = nw)

    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            #   循环 训练 + 验证 EPOCH次，默认设置240
            for epoch in range(EPOCH):
                train_loss=0.0
                train_accu=0.0
                val_loss=0.0
                val_accu=0.0

                #print(type(optimizer.param_groups[0]))
                #print(optimizer.param_groups[0]["lr"])
                #print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0
                begin=time.time()
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    length = len(train_loader)
                    inputs, labels, filename = data
                    #   将数据放入GPU/CPU
                    inputs, labels = inputs.to(device), labels.to(device)
                    #   参数梯度初始化为0
                    optimizer.zero_grad()
                    #   计算输出结果
                    outputs = net(inputs)
                    #   计算损失函数
                    loss = criterion(outputs, labels)
                    #   反向传播
                    loss.backward()
                    #   更新参数
                    optimizer.step()
                    #   调节学习率
                    #   scheduler.step()
                    if epoch >= 100 and epoch < 150:
                        LR = 0.01
                        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
                    if epoch >= 150:
                        LR = 0.001
                        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

                    #   每训练1个batch打印一次loss和准确率
                    #   得到loss张量的元素值
                    sum_loss += loss.item()
                    #   _ = 每行预测的最大值，predict = 行最大元素对应的列序号
                    _, predicted = torch.max(outputs.data, dim = 1)
                    #   获取batch labels中的元素个数
                    total += labels.size(0)
                    #   对比获取正确率
                    correct += (predicted == labels).sum()
                    #correct += predicted.eq(labels.data).cpu().sum()
                    if Debug:
                        print("[Epoch:{}/{}, Batch:{}/{}] Loss: {:.3f} | Acc: {:.3f}%".format(epoch+1,EPOCH,i+1,int(train_set.__len__()/BATCH_SIZE),sum_loss/(i+1),100.*correct/total))
                    
                    f2.write("[Epoch:{}/{}, Batch:{}/{}] Loss: {:.3f} | Acc: {:.3f}%".format(epoch+1,EPOCH,i+1,int(train_set.__len__()/BATCH_SIZE),sum_loss/(i+1),100.*correct/total))
                    f2.write('\n')
                    f2.flush()

                #   结束后打印总损失函数结果，和准确率
                #   训练时要关注loss是否收敛，值可以不管
                train_loss = sum_loss/int(train_set.__len__()/BATCH_SIZE)
                train_accu = 100.*correct/total

                #   每训练完一个epoch测试一下准确率
                #   不更新梯度
                val_accu_backup = 0
                with torch.no_grad():
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0
                    for data in test_loader:
                        net.eval()
                        images, labels, filename = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        sum_loss += loss.item()
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()

                print(test_set.__len__())
                print(VAILD_BATCH_SIZE)
                val_loss=sum_loss/int(test_set.__len__()/VAILD_BATCH_SIZE)
                #   如果后面的epoch准确率更高，将图片名和预测结果写入prediction.txt
                val_accu = 100.*correct/total
                if(val_accu >= val_accu_backup):
                    f4 = open("prediction.txt", 'w')
                    for i in range(VAILD_BATCH_SIZE):
                        f4.write("{} {}\n".format(os.path.basename(filename[i]), labels[i]))
                        val_accu_backup = val_accu
                    f4.close()

                end=time.time()
                print("[Epoch:{}/{}] Train Loss: {:.3f} | Train Acc: {:.3f}% Test Loss: {:.3f} | Test Acc: {:.3f}% Cost time:{:.2f}min".format(epoch+1,EPOCH,train_loss,train_accu,val_loss,val_accu,(end-begin)/60.0))
                print("{}".format(LR))

                writer.add_scalar("Loss/train",train_loss,epoch)
                writer.add_scalar("Loss/val",val_loss,epoch)
                writer.add_scalar("Accu/train",train_accu,epoch)
                writer.add_scalar("Accu/val",val_accu,epoch)
                writer.add_scalar("Learning rate",optimizer.param_groups[0]["lr"],epoch)

                # 将每次测试结果实时写入acc.txt文件中
                print('Saving model......')
                torch.save(net.state_dict(), '%s/net_%03d.pth' % ("./model/" , epoch + 1))
                f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch+1, val_accu))
                f.write('\n')
                f.flush()
                # 记录最佳测试分类准确率并写入best_acc.txt文件中
                if val_accu > best_acc:
                    f3 = open("best_acc.txt", "w")
                    #f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch+1,val_accu))
                    f3.close()
                    best_acc = val_accu

            print("Training Finished, TotalEPOCH=%d" % EPOCH)
