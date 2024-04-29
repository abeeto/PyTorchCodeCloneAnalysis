import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from utils import *
from dataset import MyDataset
from model import MyModel
from options import Options


def watch_dataset(data_loader):
	img , _ = next(iter(data_loader))
	img = torchvision.utils.make_grid(img).numpy()
	img = img * 255
	img = img.transpose(1,2,0).astype('uint8')
	img = Image.fromarray(img)
	img.show()


options = Options()
device = torch.device(options.args.device)
data_train = MyDataset(options, train=True)
data_test = MyDataset(options, train=False)
model = MyModel(options).to(device)

if options.args.model_load != '':
	try:
		load_model(model, options)
	except ValueError as e:
		print("Warning: %s"%str(e))
data_loader_train = torch.utils.data.DataLoader(
	dataset = data_train,
	batch_size = options.args.batch,
	shuffle = True)
data_loader_test = torch.utils.data.DataLoader(
	dataset = data_test,
	batch_size = options.args.batch,
	shuffle = True)
	
	
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss_func = nn.CrossEntropyLoss()

watch_dataset(data_loader_train)

for epoch in range(options.args.epoch):
	# train
	model.train()
	for data, label in data_loader_train:
		data, label = data.to(device), label.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = loss_func(output, label)
		loss.backward()
		optimizer.step()

		print("loss: %f"%loss.item())
		
	# test
	model.eval()
	acc = 0
	with torch.no_grad():
		for data, label in data_loader_test:
			data, label = data.to(device), label.to(device)
			output = model(data)
			
			answer = torch.argmax(output,dim=-1)
			acc += torch.sum(answer == label).item()
			
	accuracy = acc / len(data_loader_test.dataset)
	print("accuracy %.6f"%accuracy)
	save_model(model, epoch, options)

