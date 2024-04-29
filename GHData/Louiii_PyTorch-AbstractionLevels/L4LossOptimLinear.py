from helpers import *

M = lambda n, m : nn.Parameter(tc.randn(n, m)/np.sqrt(2))
V = lambda n : nn.Parameter(tc.zeros(n))

class ANN(nn.Module):
    def __init__(self, dims):
        super(ANN, self).__init__()
        tc.manual_seed(0)
        self.layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        self.prs = []
        for l in self.layers:
            self.prs += l.parameters()

    def forward(self, x):
        next_layer = x
        for l in self.layers:
            a = l(next_layer)
            next_layer = a.sigmoid()
        return next_layer/next_layer.sum(-1).unsqueeze(-1)

    def parameters(self):
        return self.prs

data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
X_tr, X_v, Y_tr, Y_v = train_test_split(data, labels, stratify=labels, random_state=0)
X_tr, X_v, Y_tr, Y_v = map(tc.tensor, (X_tr, X_v, Y_tr, Y_v))

plt.scatter(data[:,0], data[:,1], c=labels)
plt.show()

def acc(y_hat, y):
    pred = tc.argmax(y_hat, dim=1)
    return (pred==y).float().mean()

model = ANN([2, 2, 4])
print('Model Parameters:')
print(list(model.parameters()))
lr = 0.05
loss_fn = F.cross_entropy
opt = optim.SGD(model.parameters(), lr=lr)
epo = 1000
X_tr, X_v = X_tr.float(), X_v.float()
Y_tr, Y_v = Y_tr.long(), Y_v.long()

train_data = []
for i in range(len(X_tr)):
   train_data.append([X_tr[i], Y_tr[i]])

trainloader = DataLoader(train_data, batch_size=100, shuffle=True)

loss_arr = []
acc_tr, acc_v  = [], []

for e in tqdm(range(epo)):
    with tc.no_grad(): # compute validation acc
        y_hat = model(X_v)
        acc_v.append(acc(y_hat, Y_v))

    for x, y in trainloader:# batching the data
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()

        loss_arr.append(loss.item())
        acc_tr.append(acc(y_hat, y))

        opt.step()
        opt.zero_grad()

loss_acc_plot(loss_arr, acc_tr, acc_v)
