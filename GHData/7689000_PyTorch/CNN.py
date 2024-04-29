#%%

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

#%matplotlib inline

torch.manual_seed(1) # reproducible

# Hyper parameters
# train the input data n times, to save time, we just train 1 epoch
EPOCH = 1
# 50 samples at a time to pass through the epoch
BATCH_SIZE = 50
# learning rate
LR = 0.001
# set to false if you have downloaded
DOWNLOAD_MNIST = True

# MNIST digits dataset
train_data = torchvision.datasets.MNIST(
  root='./mnist/',
  # this is training data
  train=True,
  # torch.FloatTensor of shape (Color x Height x Width) and
  # normalize in the range [0.0, 1.0]
  transform = torchvision.transforms.ToTensor(),
  # download it if you don’t have it
  download = DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size()) # (60000, 28, 28)
print(train_data.train_labels.size()) # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# data loader for easy mini-batch return in training,
# the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255.
#shape from (2000, 28, 28) to (2000,1,28,28), value in range(0,1)
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential( # input shape (1,28,28)
      nn.Conv2d(
        in_channels = 1,
        out_channels = 16,
        kernel_size = 5,
        stride = 1,
        padding = 2
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(16, 32, 5, 1, 2),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.out = nn.Linear(32 * 7 * 7, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    output = self.out(x)
    return output, x

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn')

def plot_with_labels(lowDWeights, labels):
  plt.cla()
  X, Y = lowDWeights[:,0], lowDWeights[:,1]
  for x, y, s in zip(X,Y,labels):
    c = cm.rainbow(int(255 * s /9));
    plt.text(x,y,s,backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer')

plt.ion()

for epoch in range(EPOCH):
  for step, (x,y) in enumerate(train_loader):
      # gives batch data, normalize x when iterate train_loader
      b_x = Variable(x) # batch x
      b_y = Variable(y) # batch y

      output = cnn(b_x)[0]
      loss = loss_func(output, b_y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if step % 100 == 0:
        test_output, last_layer = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
        print('Epoch: ',epoch, '| train_loss: %.4f' % loss.data,
        '| test accuracy: %.2f' % accuracy)
        #print(’Epoch: ’,epoch, ’| train_loss: %.4f’ % loss.data[0],
        # ’| test accuracy: %.2f’ % accuracy)

        if HAS_SK:
        # visualization of trained flatten layer (T-SNE)
          tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
          plot_only = 500
          low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
          labels = test_y.numpy()[:plot_only]
          plot_with_labels(low_dim_embs,labels)

plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

# %%
