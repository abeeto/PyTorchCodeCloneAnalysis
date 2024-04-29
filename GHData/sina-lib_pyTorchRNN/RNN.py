import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from tqdm import tqdm

# create two simple periodic signal
N_samples = 1000
time = np.linspace(0,1.7,N_samples) + 0.2
x = np.sin( 2*np.pi* 1.8 * time)
y = [ np.exp(-np.abs(9*i)) for i in x]

# another testing:
x2 = np.sin(2*np.pi*1.0* time)
y2 = np.exp( -np.abs(9*x2) )

# plt.show()

# create an RNN:
class ZeroCross(nn.Module):
    def __init__(self):
        super(ZeroCross, self).__init__()
        self.rnn = nn.LSTM(input_size=1,hidden_size=35,num_layers=1)
        self.L1   = nn.Linear(35,1)
        self.act  = nn.Tanh()
        # self.L2   = nn.Linear(10,1)
        # self.ma   = nn.ReLU()

    def forward(self, x):
        output, hidden = self.rnn(x)
        output = self.L1(output)  # Linear 35 -> 1
        # output = self.ma(output)  # ReLU
        # output = self.L2(output)  # Linear 10 -> 1
        output = self.act(output) # Tanh
        return output

# model = nn.RNN(input_size=1,hidden_size=1,num_layers=1)
model = ZeroCross()
criterion = nn.MSELoss()
optimzer = torch.optim.Adam(model.parameters(), lr=3e-4)

# input to the RNN has the shape: (seq_len, batch, input_size)
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
N_Train = int(N_samples * 0.7)
# N_Test  = int(N_samples - N_Train)
X = torch.tensor(x, dtype=torch.float).view(N_samples,1,1)
Y = torch.tensor(y, dtype=torch.float).view(N_samples,1,1)
x_train = X[:N_Train]
y_train = Y[:N_Train]
x_test = X[N_Train:-1]
y_test = Y[N_Train:-1]
# x_test = torch.tensor(x_test).view(N_samples,-1,1)

epoch = 270
lo = []
for i in tqdm(range(epoch)):
    pred = model(x_train)
    loss = criterion(pred,y_train)
    # print(loss)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    lo.append(loss.item())



# display the input signal and zero-crossing signs
plt.plot(time,x,'r--')
plt.plot(time,y,)

# the predictions:
yy = pred[:,0,0].detach().numpy()
plt.plot(time[:N_Train],yy)
pred = model(x_test)
yy = pred[:,0,0].detach().numpy()
plt.plot(time[N_Train:-1],yy)
plt.legend(['signal', 'Ground Truth', 'model output on train', 'model output on test'])

# do another test
fig2,ax2 = plt.subplots()
ax2.plot(time,x2)
ax2.plot(time,y2)
x2 = torch.tensor(x2, dtype=torch.float).view(N_samples,1,1)
pred = model(x2).detach().numpy()
ax2.plot(time,pred[:,0,0])
plt.legend(['signal2','Ground Truth', 'model output'])

# draw loss diagram
fig,ax = plt.subplots()
ax.plot(np.array(lo))
# ax.plot.title('Loss vs epochs')


plt.show()