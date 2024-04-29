import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN_Minist_Net(nn.Module):
    #构造函数，需要继承Pytorch内的nn.Module构建网络结构
    def __init__(self):
        super(CNN_Minist_Net,self).__init__()
        self.cnn_layers = nn.Sequential(

            # in_channels —— 输入的channels数
            # out_channels —— 输出的channels数
            # kernel_size ——卷积核的尺寸，可以是方形卷积核、也可以不是，下边example可以看到
            # stride —— 步长，用来控制卷积核移动间隔
            # padding ——输入边沿扩边操作
            # padding_mode ——扩边的方式
            # bias ——是否使用偏置(即out = wx+b中的b)
            nn.Conv2d(in_channels = 1,out_channels = 8,kernel_size=3,padding= 1,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8,out_channels = 32,kernel_size=3,padding= 1,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*32,200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.LogSoftmax(dim=1)
        )
    #前向传播，构建网络运算逻辑
    def forward(self,x):
        out = self.cnn_layers(x)
        out = out.view(-1, 7 * 7 * 32)
        out = self.fc_layers(out)
        return out


#——》加载Minist数据集
# 数据预处理过程，将数据转换成Tensor，并进行归一化
transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                             ])
#加载MNIST数据集
train_ts =datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ts = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)
#——》开始训练

model = CNN_Minist_Net()
#设置优化器和定义损失函数交叉熵
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#设置opeoch的大小
epoch = 1
for iEpoch in range(epoch):
    #print("idx : %d",iEpoch)
    for idx,(x_train,y_train) in enumerate(train_dl):
         y_pred = model.forward(x_train)
         train_loss = loss_fn(y_pred,y_train)
        #前向传播完成，需要将梯度清空
         model.zero_grad()
         train_loss.backward()
         optimizer.step()
         if (idx + 1) % 100 == 0:
             print(idx + 1, train_loss.item())
#可适当简化模型
model.eval()
#预测，计算精度
total = 0;
correct_count = 0

for test_images, test_labels in test_dl:
    with torch.no_grad():
        pred_labels = model(test_images)
    predicted = torch.max(pred_labels, 1)[1]
    correct_count += (predicted == test_labels).sum()
    total += len(test_labels)
print(correct_count.cpu().detach().numpy(), total)

print("total acc : %.4f\n"%(correct_count.cpu().detach().numpy() / total))
torch.save(model, './cnn_mnist_model.pt')

#保存了模型之后，还可以转化为ONNX格式，把模型送给OpenCV DNN模块调用
dummy_input = torch.randn(1, 1, 28, 28, device='cpu')
model = torch.load("./cnn_mnist_model.pt")
torch.onnx.export(model, dummy_input, "cnn_mnist.onnx", verbose=True)      

                        
       
        









    
