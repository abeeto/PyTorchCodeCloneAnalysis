import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

batch_size = 64
train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
		self.mp = torch.nn.MaxPool2d(2)

		self.relu = torch.nn.ReLU()
		self.fc = torch.nn.Linear(320, 10)
		self.log_softmax = torch.nn.LogSoftmax()

	def forward(self, x):
		in_size = x.size(0)
		x = self.relu(self.mp(self.conv1(x)))
		x = self.relu(self.mp(self.conv2(x)))
		x = x.view(in_size, -1)
		x = self.fc(x)

		return self.log_softmax(x)

model = Network()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015, momentum=0.95)

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)

		optimizer.zero_grad()
		output = model(data)

		loss = criterion(output, target)
		loss.backward()

		optimizer.step()

		if batch_idx%10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))

def test():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		data, target = Variable(data, volatile=True), Variable(target)

		output = model(data)
		test_loss += criterion(output, target).data[0]

		pred = torch.max(output.data, 1)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10):
	train(epoch)
	test()