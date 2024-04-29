import torch as t
import numpy as np
import pandas as pd
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from model import ResNet

data = pd.read_csv('data.csv', sep=";")

training_data, testing_data = train_test_split(
    data,
    random_state=42,
    test_size=0.2,
    stratify=data[["crack", "inactive"]]
)

training_load = t.utils.data.DataLoader(
    ChallengeDataset(training_data, mode='train'),
    batch_size=10
)
testing_load = t.utils.data.DataLoader(
    ChallengeDataset(testing_data, mode='val'),
    batch_size=10
)

net = ResNet()

crit = t.nn.BCEWithLogitsLoss()

optimizer = t.optim.Adam(net.parameters(), lr=0.0001)

trainer = Trainer(
    model=net,
    crit=crit,
    optim=optimizer,
    train_dl=training_load,
    val_test_dl=testing_load,
    cuda=True,
    early_stopping_patience=100000
)

res = trainer.fit(200)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()
