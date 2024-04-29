from d2lzh_pytorch.utils import *
from collections import OrderedDict
from torch.nn import init  # PyTorch在init模块中提供了多种参数初始化方法。

batch_size = 256
num_inputs = 784
num_outputs = 10
num_epochs = 5

# 加载训练数据和测试数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# net = LinearNet2(num_inputs, num_outputs)
# 定义模型 还不含softmax函数 输出还是o  softmax集成到了损失函数中
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
# 初始化模型
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 损失函数：包括softmax运算和交叉熵损失计算的函数，PyTorch提供的函数往往具有更好的数值稳定性。
loss = nn.CrossEntropyLoss()

# 优化算法：学习率为0.1的小批量随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

