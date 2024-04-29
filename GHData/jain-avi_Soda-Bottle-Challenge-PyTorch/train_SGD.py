"""
Author : Avineil Jain 
This code is the main train file for the model. 
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models 
from cnn_architecture import CNN
from dataloader import bottle,data_split

#--------------------------------------------------------------------------
#Defining Dataloader and Transforms 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_train_transform = transforms.Compose([transforms.Resize(size = (256,256)),transforms.RandomResizedCrop(size = 224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])
soda_bottles_train = bottle('train_splt.csv', data_train_transform)
data_test_transform = transforms.Compose([transforms.Resize(size = (224,224)), transforms.ToTensor(), normalize])
soda_bottles_test = bottle('test_splt.csv', data_test_transform)
train_loader = torch.utils.data.DataLoader(dataset=soda_bottles_train, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=soda_bottles_test, batch_size=32, shuffle=True, num_workers=4)

soda_model = CNN() 
#--------------------------------------------------------------------------
#Defining Optimizers and Loss functions
loss_func = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.SGD(soda_model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
#--------------------------------------------------------------------------
#Checking for GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on ",device)
soda_model.to(device)
#----------------------------------------------------------------------------
def get_accuracy(model,dataloader):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in dataloader:
			images, labels = data
			images,labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return (100. * correct / total)

#----------------------------------------------------------------------------
#Training the Network
num_epochs = 100
epochs = num_epochs
best_acc = 0.

print("Trying SGD with Momentum and Decaying LR")
for epoch in range(epochs):
	soda_model.train()
	running_loss = 0.0
	train_correct = 0
	train_total = 0
	for i,data in enumerate(train_loader,0):
		inputs,labels = data
		inputs,labels = inputs.to(device), labels.to(device)
		#Zero the gradients 
		optimizer.zero_grad()
		outputs = soda_model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		loss = loss_func(outputs,labels)
		train_total += labels.size(0)
		train_correct += (predicted == labels).sum().item()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if (i+1) % 99 == 0:    # print every 100 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
			running_loss = 0.0

	if (epoch+1)%30==0:
		lr /= 10
		print('Decaying learning rate to:', lr)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
			print(param_group['lr'])

	train_acc = (100. * train_correct / train_total)
	print("Accuracy Training: ",train_acc)
	if(epoch%5==0):
		test_acc = get_accuracy(soda_model,test_loader)
		print("Test Accuracy: ",test_acc)
		if(test_acc >= best_acc):
			print("Accuracy Increased! Saving Model...")
			torch.save(soda_model.state_dict(), 'SGD_soda_model.pkl')
			best_acc = test_acc


print("training finshed...")
torch.save(att_model.state_dict(), 'last_SGD_soda_model.pkl')
