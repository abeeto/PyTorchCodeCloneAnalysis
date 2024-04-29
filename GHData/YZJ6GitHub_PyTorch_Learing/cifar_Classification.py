import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

train_on_gpu = torch.cuda.is_available()
#判断是否可以使用GPU
if not train_on_gpu:
    print("CUDA is not available.")
else:
    print("CUDA is  available.")
# number of subprocesses to use for data loading
num_workers = 0
 # 每批加载16张图片
batch_size = 4
 # percentage of training set to use as validation
valid_size = 0.2
# 将数据转换为torch.FloatTensor，并标准化。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )
#Loda datasets
train_data = datasets.CIFAR10('data',train=True,download= True,transform=transform)
test_data = datasets.CIFAR10('data',train=False,download= True,transform=transform)
# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size*num_train))
train_idx,valid_x = indices[split:],indices[:split]
# define samplers for obtaining training and validation batches
train_sample = SubsetRandomSampler(train_idx)
valid_sample = SubsetRandomSampler(valid_x)
# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(train_data,batch_size = batch_size,sampler = train_sample,num_workers = num_workers)
valid_loader = DataLoader(train_data,batch_size = batch_size,sampler = valid_sample,num_workers = num_workers)
test_loader = DataLoader(test_data,batch_size = batch_size,num_workers = num_workers)
# 图像分类中10类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

def imshow(img):
        img = img / 2 + 0.5  # unnormaliz
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# 获取一批样本
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# 显示图像，标题为类名
fig = plt.figure(figsize=(25, 4))
# 显示16张图片
for idx in np.arange(3):
    ax = fig.add_subplot(2, 16/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])


#定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 卷积层 (32x32x3的图像)
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2,2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4,500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500,10)
         # dropout层 (p=0.3)
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x
model = Net()
#定义损失函数和梯度下降算法
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)
#模型训练‘
n_Epochs = 10
valid_loss_min = np.Inf # track change in validation loss

for iEopch in range(1,n_Epochs):
    train_loss = 0.0
    valid_loss = 0.0
    #训练集的模型
    model.train()
    for data,target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    model.eval()
    #验证集模型
    for data,target in valid_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        valid_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(iEopch, train_loss, valid_loss))

    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

#加载模型
model.load_state_dict(torch.load('model_cifar.pt'))
#Test 
test_loss = 0.0
class_correct = list(0.for i in range(10))
class_total = list(0.for i in range(10))
model.eval()
for data,target in test_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        test_loss += loss.item()*data.size(0)
        _,pred = torch.max(output,1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


# plot the images in the batch, along with predicted and true la

#神经网络优化器优化的对象是网络的参数，因此优化器的输入参数是网络的参数
   


































