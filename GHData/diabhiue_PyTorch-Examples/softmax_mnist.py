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
		self.l1 = torch.nn.Linear(784, 512)
		self.l2 = torch.nn.Linear(512, 256)
		self.l3 = torch.nn.Linear(256, 128)
		self.l4 = torch.nn.Linear(128, 64)
		self.l5 = torch.nn.Linear(64, 32)
		self.l6 = torch.nn.Linear(32, 10)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		x = x.view(-1, 784)
		x = self.relu(self.l1(x))
		x = self.relu(self.l2(x))
		x = self.relu(self.l3(x))
		x = self.relu(self.l4(x))
		x = self.relu(self.l5(x))
		return self.l6(x)

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