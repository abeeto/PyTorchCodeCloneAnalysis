# LSTM and embedding


import torch


# 准备数据，这里没有one_hot_lookup，因为embedding优于one_hot
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3 ,1, 2 ,3, 2]

inputs = torch.LongTensor(x_data).view(seq_len, batch_size)
labels = torch.LongTensor(y_data)


# 构建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.lstm = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        batchsize = x.size(1)
        hidden = torch.zeros(num_layers, batchsize, hidden_size)
        c0 = torch.zeros(num_layers, batchsize, hidden_size)
        x = self.emb(x)
        x, _ = self.lstm(x, (hidden, c0))
        x = self.fc(x)
        return x.view(-1, num_class)


net = Model()


# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)


# 训练过程
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted:', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch+1, loss.item()))