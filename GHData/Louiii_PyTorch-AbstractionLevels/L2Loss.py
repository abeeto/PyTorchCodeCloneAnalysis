from helpers import *

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        tc.manual_seed(0)
        self.weights1 = nn.Parameter(tc.randn(2, 2)/np.sqrt(2))
        self.bias1 = nn.Parameter(tc.zeros(2))

        self.weights2 = nn.Parameter(tc.randn(2, 4)/np.sqrt(2))
        self.bias2 = nn.Parameter(tc.zeros(4))

    def forward(self, x):
        a1 = tc.matmul(x, self.weights1) + self.bias1
        h1 = a1.sigmoid()
        a2 = tc.matmul(h1, self.weights2) + self.bias2
        h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)
        return h2

data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
X_tr, X_v, Y_tr, Y_v = train_test_split(data, labels, stratify=labels, random_state=0)
X_tr, X_v, Y_tr, Y_v = map(tc.tensor, (X_tr, X_v, Y_tr, Y_v))

plt.scatter(data[:,0], data[:,1], c=labels)
plt.show()

def acc(y_hat, y):
    pred = tc.argmax(y_hat, dim=1)
    return (pred==y).float().mean()

model = ANN()
print('Model Parameters:')
print(list(model.parameters()))

loss_fn = F.cross_entropy
lr = 0.02
epo = 10000
X_tr, X_v = X_tr.float(), X_v.float()
Y_tr, Y_v = Y_tr.long(), Y_v.long()
loss_arr = []
acc_tr, acc_v  = [], []

for e in tqdm(range(epo)):
    with tc.no_grad():# compute validation acc
        y_hat = model(X_v)
        acc_v.append(acc(y_hat, Y_v))

    y_hat = model(X_tr)
    loss = loss_fn(y_hat, Y_tr)
    loss.backward()
    loss_arr.append(loss.item())
    acc_tr.append(acc(y_hat, Y_tr))

    with tc.no_grad():
        for p in model.parameters():
            p -= p.grad * lr
        model.zero_grad()

loss_acc_plot(loss_arr, acc_tr, acc_v)
