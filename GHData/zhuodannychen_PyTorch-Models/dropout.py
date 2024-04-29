import torch
import matplotlib.pyplot as plt

NUM_DATA = 50
x = torch.unsqueeze(torch.linspace(-1, 1, NUM_DATA), 1)
y = x + 0.3*torch.normal(torch.zeros(NUM_DATA, 1), torch.ones(NUM_DATA, 1))

test_x = torch.unsqueeze(torch.linspace(-1, 1, NUM_DATA), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(NUM_DATA, 1), torch.ones(NUM_DATA, 1))

plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))

overfitting = torch.nn.Sequential(
        torch.nn.Linear(1, 400),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 400),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 1),
    )

dropout = torch.nn.Sequential(
        torch.nn.Linear(1, 400),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 400),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 1),
    )

overfitting_optimizer = torch.optim.Adam(overfitting.parameters(), lr=0.01)
dropout_optimizer = torch.optim.Adam(dropout.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

plt.ion()

for epoch in range(500):
    pred_o = overfitting(x)
    pred_d = dropout(x)

    loss_o = criterion(pred_o, y)
    loss_d = criterion(pred_d, y)

    overfitting_optimizer.zero_grad()
    dropout_optimizer.zero_grad()
    loss_o.backward()
    loss_d.backward()
    overfitting_optimizer.step()
    dropout_optimizer.step()

    if epoch % 10 == 0:
        overfitting.eval()
        dropout.eval() 

        plt.cla()
        test_pred_ofit = overfitting(test_x)
        test_pred_drop = dropout(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % criterion(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % criterion(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left'); plt.ylim((-2.5, 2.5));plt.pause(0.1)

        overfitting.train()
        dropout.train()

plt.ioff()
plt.show()
