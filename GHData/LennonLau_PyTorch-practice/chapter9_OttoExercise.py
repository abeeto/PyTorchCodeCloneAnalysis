import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


# 数据处理
# 1.处理数据及加载数据集
def labelsid(labels):  # 将类别标签转换成id表示，方便以后计算交叉熵
    target_id = []
    target_labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    for label in labels:
        target_id.append(target_labels.index(label))
    return target_id


class My_dataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        labels = data['target']
        self.len = data.shape[0]
        self.x_data = torch.tensor(np.array(data)[:, 1:-1].astype(float))  # 进行强制格式转换flaot,why?
        self.y_data = labelsid(labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 2.数据实例化
filepath = r'../data/otto-group-product-classification-challenge/train.csv'
dataset = My_dataset(filepath)
# 3.设计数据集加载器
train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)


# 定义模型设计
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(93, 64)  # 93是因为一个sample有93个特征
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 9)  # 9是因为target标签有9类

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return self.linear4(x)  # softmax最后一层不用激活函数，因为这一层直接包含在交叉熵损失函数里面了

    def predict(self, x):
        with torch.no_grad():
            x = self.relu(self.forward(x))
            _, predicted = torch.max(x, dim=1)  # max返回最大的值(在前)和索引(在后)
            y = pd.get_dummies(predicted)
            return y


# 实例化模型，并建立评价函数和优化器
model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 开始训练
if __name__ == '__main__':
    for epoch in range(100):
        loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            inputs, target = data
            inputs = inputs.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            loss += loss.item()

            if batch_idx % 300 == 299:
                print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, loss/300))
                loss = 0.0

# 进行测试
test_data = pd.read_csv(r'../data/otto-group-product-classification-challenge/test.csv')
test_inputs = torch.tensor(np.array(test_data)[:, 1:].astype(float))
test_outputs = model.predict(test_inputs.float())
# 为测试集增加列标签
labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
test_outputs.columns = labels  # 为测试集的预测输出增加列标签，即为预测结果设置预测结果标签
test_outputs.insert(0, 'id', test_data['id'])  # .insect(0,'id')表示在索引值为0的这一列加入‘id’
# 保存成csv文件
out = pd.DataFrame(test_outputs)
out.to_csv(r'../data/otto-group-product-classification-challenge/predict.csv', index=False)