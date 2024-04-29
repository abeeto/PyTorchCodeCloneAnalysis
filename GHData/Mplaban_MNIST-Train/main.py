
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

# NOTE: You can only use Tensor API of PyTorch

from nnet import model

# TODO: import torch and torchvision libraries
# We will use torchvision's transforms and datasets

import torch
import torchvision
from torchvision import transforms, datasets
from random import randint
from matplotlib import pyplot as plt


# TODO: Defining torchvision transforms for preprocessing
# TODO: Using torchvision datasets to load MNIST
# TODO: Use torch.utils.data.DataLoader to create loaders for train and test
# NOTE: Use training batch size = 4 in train data loader.
train = datasets.MNIST('./data',train=True,download=True,
					  transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('./data',train=False,download=True,
					  transform=transforms.Compose([transforms.ToTensor()]))

trainset= torch.utils.data.DataLoader(train,batch_size=4,shuffle=True)
testset= torch.utils.data.DataLoader(test,batch_size=10000,shuffle=True)

# NOTE: Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.001


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# TODO: Define number of epochs
N_epoch = 40 # Or keep it as is


# TODO: Training and Validation Loop
# >>> for n epochs
## >>> for all mini batches
### >>> net.train(...)
## at the end of each training epoch
## >>> net.eval(...)

# TODO: End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)

batch_size=4

trainset=list(trainset)

for i in range(len(trainset)):
	trainset[i][0]=trainset[i][0].view(batch_size,-1)

accuracy = 0;
for epoch in range(N_epoch):
	print("Epoch ",epoch+1)
	cre_l=[]
	a=[]
	for i in range(len(trainset)):
		cressloss,acc,_=net.train(trainset[i][0],trainset[i][1],lr,False)
		cre_l.append(cressloss)
		a.append(acc)
	total_loss=sum(cre_l)/len(cre_l)
	total_acc=sum(a)/len(a)
	print('loss: ', total_loss)
	print('accuracy: ', total_acc)
torch.save(net,'model.pt')
	
#TESTING MODEL

batch_size_test=10000

test_loader=list(testset)


for i in range(len(test_loader)):
	test_loader[i][0]=test_loader[i][0].view(batch_size_test,-1)


for i in range(len(test_loader)):
	net.eval(test_loader[i][0],test_loader[i][1])


#PREDICTIONS FROM TRAINED MODEL
predict_loader = torch.utils.data.DataLoader(test,batch_size=10, shuffle=True)

batch_size_predict=10

predict_loader=list(predict_loader)


for i in range(len(predict_loader)):
	predict_loader[i][0]=predict_loader[i][0].view(batch_size_predict,-1)

a=randint(0,len(predict_loader))
prediction_v,pred=net.predict(predict_loader[a][0])
print(pred)
