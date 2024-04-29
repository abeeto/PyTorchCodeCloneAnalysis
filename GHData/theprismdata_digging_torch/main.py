import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt

#Simple network
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


def classification(path):
    try:
        dataset = pd.read_csv(path)
        label_map = {}
        for idx, label in enumerate(dataset['variety'].unique()):
            label_map[label] = idx
        dataset["variety"] = dataset["variety"].apply(lambda x: label_map[x])
        X = dataset.drop("variety", axis=1).values
        y = dataset["variety"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) #converto 0 to 1

        # Split the data set into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

        model = Model(X_train.shape[1])
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        loss_fn = nn.CrossEntropyLoss()

        EPOCHS = 100
        loss_list = np.zeros((EPOCHS,))
        accuracy_list = np.zeros((EPOCHS,))

        for epoch in tqdm.trange(EPOCHS):
            y_pred = model.forward(X_train)
            loss = loss_fn(y_pred, y_train)
            loss_list[epoch] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = model(X_test)
                correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
                accuracy_list[epoch] = correct.mean()
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

        ax1.plot(accuracy_list)
        ax1.set_ylabel("validation accuracy")
        ax2.plot(loss_list)
        ax2.set_ylabel("validation loss")
        ax2.set_xlabel("epochs");
        plt.show()
        print('Finished')
    except Exception as e:
        print(str(e))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    classification('dataset/iris.csv')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
