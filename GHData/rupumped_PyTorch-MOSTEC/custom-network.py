import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(3)

# Make data
N_DATA = 10000
x = 4*torch.rand(N_DATA,1)-2
y = torch.abs(x)

# Initialize Model
DIM_INPUT = 1
DIM_OUTPUT = 1
H = 2

class MyNet(torch.nn.Module):
    def __init__(self) -> None:
        super(MyNet, self).__init__()
        self.m = torch.nn.Linear(1,2,bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        s = self.m(x)
        z = self.relu(s)
        # breakpoint()
        y = torch.sum(z,1)
        return y

network = MyNet()

# Split dataset
N_train = int(3*len(y)/5)
dataset = torch.utils.data.TensorDataset(x, y)
train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [N_train,len(y)-N_train])

# Construct dataloaders for batch processing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32)
val_loader   = torch.utils.data.DataLoader(dataset=val_dataset  , batch_size=32)

# Define learning hyperparameters
loss_fn       = torch.nn.MSELoss(reduction='sum')
learning_rate = .00001
n_epochs      = 2000
optimizer     = torch.optim.Adam(network.parameters(), lr=learning_rate)

# Initialize arrays for logging
training_losses = []
validation_losses = []

# Main training loop
try:
    for t in range(n_epochs):
        # Validation
        with torch.no_grad():
            losses = []
            for x, y in val_loader:
                y_hat = network(x)
                loss = loss_fn(y,y_hat)
                losses.append(loss.item())
            validation_losses.append(np.mean(losses))

        # Terminating condition
        if t>20 and np.mean(validation_losses[-10:-6])<=np.mean(validation_losses[-5:-1]):
            break

        # Training
        losses = []
        for x, y in train_loader:
            y_hat = network(x)
            loss = loss_fn(y,y_hat)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_losses.append(np.mean(losses))

        pstr = f"[{t+1}] Training loss: {training_losses[-1]:.3f}\t Validation loss: {validation_losses[-1]:.3f}"

        print(pstr)
except KeyboardInterrupt:
    print('Stopping due to keyboard interrupt. Save and continue (Y/N)?')
    ans = input()
    if ans[0].upper() == 'N':
        exit(0)

fig, axs = plt.subplots(1,2)
axs[0].semilogy(range(len(  training_losses)),   training_losses, label=  'Training Loss')
axs[0].semilogy(range(len(validation_losses)), validation_losses, label='Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(x,y,'.', label='Data')
xs = torch.tensor([[-2],[-1],[0],[1],[2]]).type(torch.FloatTensor)
ys = network(xs)
axs[1].plot(xs.detach().numpy(),ys.detach().numpy(), label='Model')
axs[1].legend()
plt.show()

for name, param in network.named_parameters():
    if param.requires_grad:
        print(name, param.data)