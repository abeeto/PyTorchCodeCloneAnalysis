import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# 数据预处理过程，将数据转换成Tensor，并进行归一化
transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                             ])
#加载MNIST数据集
train_ts =datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ts = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)
#利用keras构建模型
model = torch.nn.Sequential(
   torch.nn.Linear(784, 200),
   torch.nn.ReLU(),
    torch.nn.Linear(200, 100),
   torch.nn.ReLU(),
   torch.nn.Linear(100, 10),
   torch.nn.LogSoftmax(dim=1)
)
#设置优化器和定义损失函数
loss_fn = torch.nn.NLLLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#前向传播
epoch = 20;
for s in range(epoch):
   print("run in step : %d"%s)
   for i, (x_train, y_train) in enumerate(train_dl):
       x_train = x_train.view(x_train.shape[0], -1)     
       y_pred = model(x_train)
       train_loss = loss_fn(y_pred, y_train)
       if (i + 1) % 100 == 0:
           print(i + 1, train_loss.item())
       #前向传播完成，需要将梯度清空
       model.zero_grad()                   
       train_loss.backward()
       optimizer.step()
#预测，计算精度
total = 0;
correct_count = 0
for test_images, test_labels in test_dl:
   for i in range(len(test_labels)):
       image = test_images[i].view(1, 784)
       with torch.no_grad():
           pred_labels = model(image)
       plabels = t.exp(pred_labels)
       #数据类型转换：将tensor转换成numpy-->List
       probs = list(plabels.numpy()[0])
       pred_label = probs.index(max(probs))
       true_label = test_labels.numpy()[i]
       if pred_label == true_label:
           correct_count += 1
       total += 1
print("total acc : %.2f\n"%(correct_count / total))
#模型保存
t.save(model, './nn_mnist_model.pt')