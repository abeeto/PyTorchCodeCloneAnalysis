import torch
import torch.autograd as autograd
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
import argparse
import os
'''
def imshow(img):
	img=img+0.5
	npimg=img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.show()
'''

#images,classes=next(iter(trainLoader))
#img=torchvision.utils.make_grid(images,nrow=32)
#imshow(img)
#define the model

parser=argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--resume',default='',type=str,metavar='PATH',
					help='path to latest checkpoint(default: none)')
parser.add_argument('--epochs',default='',type=int,metavar='N',\
					help='number of total epochs to run')
#MLP
class MLP(torch.nn.Module):
	def __init__(self):
		super(Network,self).__init__()
		self.fc1=torch.nn.Linear(28*28,500)
		self.fc2=torch.nn.Linear(500,256)
		self.fc3=torch.nn.Linear(256,10)
	def forward(self,x):
		x=x.view(-1,28*28)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=F.relu(self.fc3(x))
		return x
#LeNet
class LeNet(torch.nn.Module):
	def __init__(self):
		super(LeNet,self).__init__()
		self.conv1=torch.nn.Conv2d(1,20,5,1)
		self.conv2=torch.nn.Conv2d(20,50,5,1)
		self.fc1=torch.nn.Linear(50*4*4,500)
		self.fc2=torch.nn.Linear(500,10)
	def forward(self,x):
		x=self.conv1(x)
		x=F.max_pool2d(x,2,2)
		x=F.relu(x)
		x=self.conv2(x)
		x=F.max_pool2d(x,2,2)
		x=F.relu(x)
		x=x.view(-1,4*4*50)
		x=self.fc1(x)
		x=self.fc2(x)
		return x




def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
	torch.save(state,filename)
	if is_best:
		shutil.copyfile(filename,'model_best.pth.tar')


#define the train process
def train(model,trainLoader,optimizaton,loss_function,epoch):
	Loss=[]
	for batch_idx,(x,target) in enumerate(trainLoader):
		optimizaton.zero_grad()
		x,target=Variable(x),Variable(target)
		pre_target=model(x)
		loss=loss_function(pre_target,target)			
		Loss.append(loss.data[0])
		loss.backward()
		optimizaton.step()
		'''
		if batch_idx%100==0:
			print("epoch: %d,batch index:  %d,train loss: %.6f"\
				%(epoch,batch_idx,loss.data[0]))
		'''
	return Loss
#define the validationn function
def test(model,testLoader,batch_size):
	correct_cnt=0
	for batch_idx,(x,target) in enumerate(testLoader):
		x,target=Variable(x,volatile=True),Variable(target,volatile=True)
		score=model(x)
		_,pred_label=torch.max(score.data,1)
		correct_cnt+=(pred_label==target.data).sum()		
	accuracy=correct_cnt*1.0/len(testLoader)/batch_size
	return accuracy
def main():
	#define the model and pamameters
	global args
	args=parser.parse_args()
	model=LeNet()
	loss_function=torch.nn.CrossEntropyLoss()
	optimizaton=torch.optim.SGD(model.parameters(),lr=0.1)

	AlossData=[]
	best_accuracy=0
	

	#Data loding code
	trans=transforms.Compose([transforms.ToTensor(),
	                      transforms.Normalize((0.5,),(1.0,))])
	train_all_mnist=datasets.MNIST('./data',train=True,transform=trans,\
	             download=False)
	test_all_mnist=datasets.MNIST('./data',train=False,transform=trans)
	batch_size=128
	trainLoader=torch.utils.data.DataLoader(train_all_mnist,batch_size)
	testLoader=torch.utils.data.DataLoader(test_all_mnist,batch_size,shuffle=False)
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint=torch.load(args.resume)
			model.load_state_dict(checkpoint['state_dict'])
			optimizaton.load_state_dict(checkpoint['optimizer'])
			breakAccur=test(model,testLoader,batch_size)
	else:
		print("=>  no checkpoint found at '{}'".format(args.resume))
		for epoch in range(args.epochs):		
			AlossData.extend(train(model,trainLoader,optimizaton,loss_function,epoch))
			accuracy=test(model,testLoader,batch_size)	
			print("epoch:'{epoch}',Test accuray:'{accuracy}'".format(epoch=epoch,accuracy=accuracy))
			if best_accuracy<accuracy:
				best_accuracy=accuracy
				save_checkpoint({
				'epoch':epoch+1,
				'best_accuracy':best_accuracy,
				'state_dict':model.state_dict(),
				'optimizer':optimizaton.state_dict(),
				 },1)
		breakAccur=best_accuracy
	'''
	fig=plt.figure()	
	ax=fig.add_subplot(111)
	batch=[i for i in range(len(AlossData))]
	ax.plot(batch,AlossData,'r-')
	plt.show(block=False)
	'''
	print('accuracy of test dataset:',breakAccur)

if __name__=='__main__':
	main()
'''
the_model=TheModelClass(*args,**kwargs)
the_model.load_state_dict(torch.load('./data'))
'''



