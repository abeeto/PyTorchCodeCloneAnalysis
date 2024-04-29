import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# gpu的配置
device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

# 超参数的设定
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
                                          
# 模型搭建
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Neuralnet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn,Linear(hidden_size, num_classes)
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

# 创建模型并转移至gpu        
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # forward 计算
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # tensor.item()  获取tensor的数值
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
# 测试model
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1)
        total+=labels.size(0) # labels numpy的array类型，0代表第0维，size(0)代表行数
        correct += (predicted == labels).sum().item()
        
     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# 保存模型
torch.save(model.state_dict, 'model.ckpt')
        
        
        
