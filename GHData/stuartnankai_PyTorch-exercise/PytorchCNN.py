import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


"""
Building step:
1. Put the data into the Data loader
2. Bulid the NN class, super(CNN, self).__init__() is necessary,  using __init__() to build the structure of NN and
using forward() to transfer the data, finally return the output.
3. Define the optimizer
4. Define the loss function
5. For loop to train the NN
5.1 the input x, y need to be put into the Variable
5.2 the optimizer need to clean the gradient by using zero_grad()
5.3 BP
5.4 optimizer.step()
6. Model can be saved as pkl or loaded by using torch.save or torch.load
"""

# hyper parameters

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
LOADMODEL = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# plot

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
#
# plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)

#
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.test_labels[:2000]


# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1, 28,28)
                in_channels=1,
                kernel_size=5,
                out_channels=16,
                stride=1,
                padding=2  # if stride = 1, padding = (kernal_size-1)/2 = (5-1)/2 = 2
            ),  # -> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2  # -> (16,14,14)
            ),
        )
        self.conv2 = nn.Sequential(  # (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> (32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 10 numbers

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7,7)
        x = x.view(x.size(0), -1)  # (batch, 32* 7 * 7 )  view: flat the data
        output = self.out(x)

        return output,x


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()




for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # test_output = cnn(test_x)
            # pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # print("This is pre_y:", pred_y.size())
            # print("This is test_y:", pred_y.size())
            # print("This is test_y.size(0):", test_y.size(0))
            # print(sum(pred_y==test_y).__int__())
            # hit_num = sum(pred_y==test_y).__int__()
            # accuracy = hit_num / test_y.size(0)
            # print("Epoch:", epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: ', accuracy)

            # test_output= cnn(test_x)
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'Prediction number')
print(test_y[:10].numpy(), ' Real number')

torch.save(cnn,'cnn.pkl')
