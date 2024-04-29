import torch
import numpy as np
import matplotlib.pyplot as plt

from FFNet import FFNet

num_hidden = 15
num_runs = 50
num_epochs = 500

x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y_train = torch.tensor([0, 1, 1, 0], dtype=torch.float)

loss_history = np.zeros((num_runs, num_epochs))

for run in range(num_runs):
    loss_run = 0

    model = FFNet(2, num_hidden, 1)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        loss_epoch = 0
        for sample in range(len(x_train)):
            x = x_train[sample % len(x_train)]
            y = y_train[sample % len(x_train)]
            y_prediction = model(x)

            loss = criterion(y_prediction, y)
            loss_epoch += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_history[run, epoch] = loss_epoch / len(x_train)

    print("Run", run + 1, "Complete. End Loss:", loss_history[run, -1])

for run in range(num_runs):
    plt.plot(range(num_epochs), loss_history[run])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss During Training over " + str(num_runs) + " Runs.")
plt.show()
