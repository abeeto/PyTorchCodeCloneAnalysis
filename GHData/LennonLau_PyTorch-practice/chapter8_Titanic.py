import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


lr = 0.002
epochs = 100
predict_need = True


class TitanicData(Dataset):
    def __init__(self, filepath):
        # 不取‘Age‘是因为’Age‘有的sample缺省，干脆就不取了
        x_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
        y_features = ['Survived']

        data = pd.read_csv(filepath)
        self.len = data.shape[0]

        # get_dummies是one_hot编码，实际上x_data的维度从6变成了7
        # x和y都采取data[features]的形式是为了使y也是矩阵，与x的shape相同
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(data[x_features])))
        self.y_data = torch.from_numpy((np.array(data[y_features])))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


filepath = r'../data/titanic/train.csv'
train_set = TitanicData(filepath)
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=0)


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(6, 3)
        self.linear2 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            print(x.shape)
            y = []
            for i in x:
                y.append(1 if i >= 0.5 else 0)
            return y


model = LogisticRegression()


criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


def show_lossimage(epoch_list, loss_list):
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    epoch_list = []
    loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.float()  # 要进行数据类型转换，否则会报错，但是是为什么呢？
            y = y.float()
            y_hat = model(x)

            loss = criterion(y_hat, y)

            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
        epoch_loss /= (i+1)
        print('epoch={}, loss={}'.format(epoch+1, epoch_loss))
        epoch_list.append(epoch+1)
        loss_list.append(epoch_loss)

    show_lossimage(epoch_list, loss_list)

    # predict
    if predict_need:
        # prepare the test dataset
        test_filepath = r'../data/titanic/test.csv'
        test_data = pd.read_csv(test_filepath)
        x_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
        # 数据类型经历df, np.ndarray, tensor
        x = torch.from_numpy(np.array(pd.get_dummies(test_data[x_features])))

        # get the prediction
        y_pred = model.predict(x.float())

        # save the prediction in csv
        outputs = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
        outputs.to_csv(r'../TitanicPredict.csv', index=False)  # False表示不保存索引