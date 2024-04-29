import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data

torch.manual_seed(1024)    # 固定随机数种子，用于再现结果
# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )
'''
torch.normal(means=torch.arange(1, 11), std=torch.arange(1, 0, -0.1))
 1.5104#是从均值为1，标准差为1的正态分布中随机生成的
 1.6955#是从均值为2，标准差为0.9的正态分布中随机生成的
 2.4895
 4.9185
 4.9895
 6.9155
 7.3683
 8.1836
 8.7164
 9.8916
'''
# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# # 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

batch_size = 100
hidden_dim = 10

# 先转换成 torch 能识别的 Dataset

torch_dataset = Data.TensorDataset(x, y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               
    num_workers=2,              # 多线程来读数据
)


import torch
import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义Graph
        self.input = torch.nn.Linear(2, 10)   # 输入层
        self.hidden = torch.nn.Linear(10, hidden_dim)   # 隐藏层
        self.predict = torch.nn.Linear(hidden_dim, 2)  # 输出层
        self.att_weights = torch.autograd.Variable(torch.randn(batch_size,1,hidden_dim),requires_grad=True) # query
        self.MLP = torch.nn.Sequential(torch.nn.Linear(hidden_dim,32),torch.nn.ReLU(),torch.nn.Linear(32,hidden_dim))

    def attention(self,encoder_outputs):
        encoder_outputs = encoder_outputs.unsqueeze(1) # (batch_size,timestep=1,hidden_dim)
        M = torch.tanh(encoder_outputs) # (batch_size,timestep=1,hidden_dim)
        alpha = torch.bmm(self.att_weights,M.transpose(1,2)) # (batch_size,1,timestep=1)
        softmax_alpha = F.softmax(alpha,dim=-1) # (batch_size,1,timestep=1); timestep=1时，softmax->sigmoid? 如果没有概率分布了，那么attention的意义是什么？
        context_vector = torch.bmm(softmax_alpha,encoder_outputs) # (batch_size,timestep=1,hidden_dim)
        context_vector = context_vector.squeeze(1) # (batch_size,hidden_dim)
        return context_vector
    
    def self_attention(self,encoder_outputs):
        encoder_outputs = encoder_outputs.unsqueeze(1) # (batch_size,timestep=1,hidden_dim)
        query = self.MLP(encoder_outputs) # (batch_size, timestep=1, hidden_dim)
        alpha = torch.bmm(query,torch.tanh(encoder_outputs).transpose(1,2)) # (batch_size,1,timestep=1)
        softmax_alpha = F.softmax(alpha,dim=-1) # (batch_size,1,timestep=1); timestep=1时，softmax->sigmoid? 如果没有概率分布了，那么attention的意义是什么？
        context_vector = torch.bmm(softmax_alpha,encoder_outputs) # (batch_size,timestep=1,hidden_dim)
        context_vector = context_vector.squeeze(1) # (batch_size,hidden_dim)
        return context_vector



    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.input(x))      
        x = F.relu(self.hidden(x)) 
        # x = self.attention(x)
        # x = self.self_attention(x)
        x = self.predict(x)             # 输出值
        return x

net = Net()

# print(net)  # net 的结构
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.GELU(),
    torch.nn.Linear(10, 10),
    torch.nn.GELU(),
    torch.nn.Linear(10, 2)
)


# optimizer 是训练的工具
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()
def train():
    for epoch in range(10):
        for step, (batch_x, batch_y) in enumerate(loader):
            # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())
            out = net(batch_x)     # 喂给 net 训练数据 x, 输出分析值

            loss = loss_func(out, batch_y)     # 计算两者的误差

            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

            if step % 2 == 0:
                plt.cla()
                # 过了一道 softmax 的激励函数后的最大概率才是预测值
                tmp = F.softmax(out) # (200,2)
                # prediction = torch.max(F.softmax(out), 1)[1]
                prediction = torch.max(tmp,1) # 
                
                '''
                torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
                Returns a namedtuple (values, indices) where 
                values is the maximum value of each row of the input tensor in the given dimension dim. 
                And indices is the index location of each maximum value found (argmax).
                '''
                prediction = prediction[1] # prediction[0]是原始概率值的向量，prediction[1]是argmax之后的向量，shape都是(200,)
                pred_y = prediction.data.numpy()
                pred_y = pred_y.squeeze()
                '''
                np.squeeze() 从数组的形状中删除单维，即把shape中为1的维度去掉
                x = np.array([[[0], [1], [2]]]) (1,3,1)
                np.squeeze(x) (3,)
                    [0 1 2]
                '''
                target_y = batch_y.data.numpy()
                plt.scatter(batch_x.data.numpy()[:, 0], batch_x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
                accuracy = sum(pred_y == target_y)/(batch_size)  # 预测中有多少和真实值一样
                print(accuracy)
                # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
                plt.text(0.0, 0.0, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
                plt.pause(0.1)
    
    out = net(x[20:20+batch_size])
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = y.data.numpy()[20:20+batch_size]
    accuracy = sum(pred_y == target_y)/(batch_size)
    plt.cla()
    plt.scatter(x[20:20+batch_size].data.numpy()[:, 0], x[20:20+batch_size].data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
    plt.text(0.0, 0.0, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
    plt.pause(0.1)
    print("final accuracy:",accuracy)
    plt.ioff()  # 停止画图
    plt.show()
    torch.save(net, 'net.pkl')  # 保存整个网络
    torch.save(net.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)

if __name__ == '__main__':
    train()