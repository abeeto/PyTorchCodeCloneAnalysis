import os
import time

import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib import cm

torch.manual_seed(1)

# Hyper Params
EPOCH = 24
BATCH_SIZE = 64
LR = 1e-3
Download_Minist = False
# Minist digits dataset
if not (os.path.exists('./mnist')) or not os.listdir('./mnist/'):
	Download_Minist = True
train_data = torchvision.datasets.MNIST(
	root='./mnist.',
	train=True,
	transform=torchvision.transforms.ToTensor(),  # PIL image to FloatTensor
	download=Download_Minist,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255
test_x = test_x.cuda()
#  volatile – 布尔值，指示这个Variable是否被用于推断模式(即，不保存历史信息)。更多细节请看Excluding subgraphs from backward。只能改变leaf variable的这个标签。
test_y = test_data.test_labels[:2000].cuda()


def plot_with_labels(lowDWeights, labels):
	plt.cla()
	X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
	for x, y, s in zip(X, Y, labels):
		c = cm.rainbow(int(255 * s / 9))
		plt.text(x, y, s, backgroundcolor=c, fontsize=9)
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.title('Visualize last layer')
	plt.show()
	plt.pause(0.01)


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv1 = nn.Sequential(  # container
			nn.Conv2d(
				in_channels=1,
				out_channels=16,
				kernel_size=5,
				stride=1,
				padding=2,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.out = nn.Linear(32 * 7 * 7, 10)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		output = self.out(x)
		return output, x


cnn = CNN()
cnn.cuda()
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
loss_func.cuda()

try:
	from sklearn.manifold import TSNE
	
	HAS_SK = True
except:
	HAS_SK = False;
	print('Install sklearn please')

plt.ion()

# training and testing

for epoch in range(EPOCH):
	for step, (x, y) in enumerate(train_loader):
		b_x = Variable(x)
		b_x = b_x.cuda()
		b_y = Variable(y)
		b_y = b_y.cuda()
		b_x = b_x.type(torch.cuda.FloatTensor)
		b_y = b_y.type(torch.cuda.LongTensor)
		output = cnn(b_x)[0]
		loss = loss_func(output, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		t = time.time()
		if step % 50 == 0:
			test_output, last_layer = cnn(test_x)
			pred_y = torch.max(test_output, 1)[1].data.squeeze()
			accuracy = float(sum(pred_y == test_y)) / float(test_y.size(0))
			print('Epoch', epoch, '| train loss: %.4f' % loss.data[0], '|test accuracy: %.2f' % accuracy)
			# if HAS_SK:
			# 	# Visualization of trained flattern(T-SNE)
			# 	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
			# 	plot_only = 500
			# 	low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
			# 	labels = test_y.cpu().numpy()[:plot_only]
			# 	plot_with_labels(low_dim_embs, labels)
			print(time.time() - t)
			t = time.time()
	
# plt.ioff()
torch.save(cnn, 'minist_model.pkl')


def data_set_view():
	print(train_data.train_data.size())
	print(train_data.train_labels.size())
	
	plt.imshow(train_data.train_data[1].numpy(), cmap='gray')
	plt.title('%i' % train_data.train_labels[1])
	plt.show()
"""
Epoch 0 | train loss: 0.2065 |test accuracy: 0.07
0.11469340324401855
Epoch 0 | train loss: 0.1249 |test accuracy: 0.06
0.11369848251342773
Epoch 0 | train loss: 0.0841 |test accuracy: 0.07
0.12367081642150879
Epoch 0 | train loss: 0.0415 |test accuracy: 0.07
0.11469244956970215
"""