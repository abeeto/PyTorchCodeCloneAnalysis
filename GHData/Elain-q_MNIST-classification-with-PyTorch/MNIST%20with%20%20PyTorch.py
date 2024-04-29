import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms,datasets
transformer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
#transforms.ToTensor将PILImage转化为torch.FloatTensor形式,Normalize a = (a-mean)/std
train_dataset = datasets.MNIST('data/',train = True,transform = transformer,download = True)
test_dataset = datasets.MNIST('data/',train = False,transform = transformer,download = True)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 32,shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 32,shuffle = True)

class net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1,10,kernel_size = 5)
		self.conv2 = nn.Conv2d(10,20,kernel_size = 5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320,50)
		self.fc2 = nn.Linear(50,10)

	def forward(self,x):
		x = F.relu(F.max_pool2d(self.conv1(x),2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
		x = x.view(-1,320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x,training = self.training)
		#training用于记录是否在训练过程中
		x = self.fc2(x)
		return F.log_softmax(x)

def fit(epoch,model,data_loader,phase = 'training',volatile = False):
	#volatile = False需要实时更新梯度
	if phase == 'training':
		model.train()
	if phase == 'validation':
		model.eval()
		volatile = True
	#通过model.对Batch normalization和dropout进行自动设置
	running_loss = 0.0
	running_correct = 0
	for batch_idx,(data,target) in enumerate(data_loader):
		if torch.cuda.is_available():
			data,target = data.cuda(),target.cuda()
		data,target = Variable(data,volatile),Variable(target)
		if phase == 'training':
			optimizer.zero_grad()
		#每个batch导数清零，但是test的时候就完全不需要因为并没有进行broadcast
		output = model(data)
		loss = F.nll_loss(output,target)
		running_loss += F.nll_loss(output,target,size_average = False).data
		preds = output.data.max(dim = 1,keepdim = True)[1]
		running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
		if phase == 'training':
			loss.backward()
			optimizer.step()
	loss = running_loss/len(data_loader.dataset)
	accuracy = 100. *running_correct/len(data_loader.dataset)
	print(loss, accuracy)
	return loss,accuracy

model = net()
if torch.cuda.is_available():
	model.cuda()
optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum = 0.5)
#model.parameters()是进行优化的对象，Variable类型的变量
train_losses,train_accuracy = [],[]
val_losses,val_accuracy = [],[]
for epoch in range(1,20):
	epoch_loss,epoch_accuracy = fit(epoch,model,train_loader,phase = 'training')
	val_epoch_loss,val_epoch_accuracy = fit(epoch,model,test_loader,phase = 'validation')
	train_losses.append(epoch_loss)
	train_accuracy.append(epoch_accuracy)
	val_losses.append(val_epoch_loss)
	val_accuracy.append(val_epoch_accuracy)