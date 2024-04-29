# Exercise 13-2 - Create Image Captioning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np

batch_size = 60
test_size = 6

#Define number of layers for each resnet block
layers = [3, 4, 6, 4]

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=False)
args = parser.parse_args()

#Run on MNIST to trial - Although ResNet designed for larger images
#use MNIST as a small toy-test
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_size,
					  shuffle=False)


num_classes = 13
hidden_size = 200  # output from the LSTM. 5 to directly predict one-hot
sequence_length = 4  
encode_size = 200
image_size = 512

ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'This', 'is', 'a']


#Use Resnet layout from previous exercises for the CNN layers
#Create a Block for two convolutional segments (where residual connections form)
class Block(nn.Module):
    def __init__(self, numIn, numOut, stride=1):
	super(Block, self).__init__()
	self.stride = stride

	#only apply stride to first for reduction at start of each connection
	#Conv and Batch_norm - no need for dropout
	self.conv1 = nn.Conv2d(numIn, numOut, kernel_size=3, stride=stride, padding=1)
	self.b_norm1 = nn.BatchNorm2d(numOut)
	self.conv2 = nn.Conv2d(numOut, numOut, kernel_size=3, padding=1)
	self.b_norm2 = nn.BatchNorm2d(numOut)

	#Define the connection of residual (unit or reduction) 
	self.convr, self.b_normr = self.get_res(numIn, numOut)


    def forward(self, x):
	#Define unit residual connection
	r = x

	#First Layer
	x = self.conv1(x)
	x = F.relu(self.b_norm1(x))
	
	#Second Layer
	x = self.conv2(x)	
	x = self.b_norm2(x)

	#If dimensionality changes perform reduction
	if self.convr != None:
	    r = self.b_normr(self.convr(r))
	
	#Add Residual
	x += r

	#Activation
	x = F.relu(x)
	return x

    def get_res(self, numIn, numOut):
	if numIn == numOut:
	    return None, None
	else:
	    return nn.Conv2d(numIn, numOut, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(numOut)



class Resnet(nn.Module):
    def __init__(self, bn, layers, image_size):
	super(Resnet, self).__init__()
	#Define residual/number of repeats
	self.residual = None
	self.layers = layers
	self.image_size = image_size

	#Initial convolution

	#Stride changed for mnist - original resnet stride=2
	#Input size 1 for MNIST, 3 for rgb (Imagenet, etc)
	self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=1)
	self.b_norm1 = nn.BatchNorm2d(64)
	self.mp1 = nn.MaxPool2d(2)

	#First Layers
	self.conv2 = []
	for i in range(layers[0]):
	    self.conv2.append(Block(64, 64))
	self.conv2 = nn.ModuleList(self.conv2)

	#Second Layers
	self.conv3 = []
	self.conv3.append(Block(64, 128, stride=2))
	for i in range(layers[1]-1):
	    self.conv3.append(Block(128, 128))
	self.conv3 = nn.ModuleList(self.conv3)

	#Third Layers
	self.conv4 = []
	self.conv4.append(Block(128, 256, stride=2))
	for i in range(layers[2]-1):
	    self.conv4.append(Block(256, 256))
	self.conv4 = nn.ModuleList(self.conv4)

	#Fourth Layers
	self.conv5 = []
	self.conv5.append(Block(256, 512, stride=2))
	for i in range(layers[3]-1):
	    self.conv5.append(Block(512, 512))
	self.conv5 = nn.ModuleList(self.conv5)

	#Final
	self.fc = nn.Linear(512, image_size)
	

    def forward(self, x):
	x = self.conv1(x)
	x = F.relu(self.b_norm1(x))
	
	# Remove for mnist - reduced dimensions
	#x = self.mp1(x)

	for i in range(self.layers[0]):
	    x = self.conv2[i](x)
	
	for i in range(self.layers[1]):
	    x = self.conv3[i](x)

	for i in range(self.layers[2]):
	    x = self.conv4[i](x)

	for i in range(self.layers[3]):
	    x = self.conv5[i](x)

	#Final average pooling
	x = F.avg_pool2d(x, kernel_size=3, padding=1)

	x = x.view(-1, 512)

	return x




#Use a simple decoder RNN
#First input is output of CNN
#All other inputs are the embedding of the previous output
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, sequence_length, encode_size, image_size, num_classes):
	super(DecoderRNN, self).__init__()
	self.hidden_size = hidden_size
	self.sequence_length = sequence_length
	self.encode_size = encode_size
	self.image_size = image_size
	self.num_classes = num_classes

	#For embedding
	self.emb = nn.Embedding(self.num_classes, hidden_size)

	#For the input of the first CNN features
	self.input = nn.Linear(self.image_size, hidden_size)
	self.b_norm = nn.BatchNorm2d(hidden_size)

	#RNN
	self.gru = nn.GRU(hidden_size, hidden_size)

	#Final linear layer
	self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
	self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, y, h, it):
	#Decide whether CNN features of embedding are input
	if it != 0:
	     s = self.emb(y)
	else:
	     s = self.input(y)
	     s = self.b_norm(s)
	     s = F.relu(s)
	     s = s.unsqueeze(1)

	#RNN and choose output
	y, h = self.gru(s.transpose(0, 1), h.transpose(0, 1))
	y = y.transpose(0, 1)
	h = h.transpose(0, 1)

	y = F.relu(self.fc1(h))

	y = F.log_softmax(self.fc2(y))
	
	return y, h	


model = Resnet(64, layers, image_size)
if args.cuda:
    model.cuda()

dec = DecoderRNN(hidden_size, sequence_length, encode_size, image_size, num_classes)
if args.cuda:
    dec.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(dec.parameters()), lr=0.0005)

##############################
#Generate test 'This is a'   #
##############################

words = [10, 11, 12]
words = torch.LongTensor(words)
words_batch = []
for i in range(batch_size):
    words_batch.append(words)
words = torch.stack(words_batch)

words_t = [10, 11, 12]
words_t = torch.LongTensor(words_t)
words_t_batch = []
for i in range(test_size):
    words_t_batch.append(words_t)
words_t = torch.stack(words_t_batch)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
	#Perform CNN operation using RESNET
	target = torch.cat([words, target.view(-1, 1)], 1)

	if args.cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
	else:
            data, target = Variable(data), Variable(target)
        output = model(data)
	#Feed output as hidden state into text generator
	if args.cuda:
	    h = torch.zeros(batch_size, 1, hidden_size)
	    h = Variable(h.cuda())
	else:
	    h = Variable(torch.zeros(batch_size, 1, hidden_size))	    
	outputs = []
	#Iterate for length of sequence
	for i in range(sequence_length):
	    if i > 0:
	        _, idx = y.transpose(1, 2).max(1)
	 	y = Variable(idx.data.view(-1, 1).cuda())
	    else:
	        y = output
	    y, h = dec(y, h, i)
	    outputs.append(y.transpose(0, 1)[0])
	output = torch.stack(outputs).transpose(0, 1)

	#Perform backprop
	model.zero_grad()
	dec.zero_grad()
	loss = 0

	for i in range(batch_size):
            loss += (criterion(output[i], target[i]))/batch_size
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    for data, target in test_loader:
	if count > 5:
	    break
	count+=1
	target = torch.cat([words_t, target.view(-1, 1)], 1)

	if args.cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
	else:
            data, target = Variable(data), Variable(target)
        output = model(data)
	#Feed output as hidden state into text generator
	if args.cuda:
	    h = torch.zeros(test_size, 1, hidden_size)
	    h = Variable(h.cuda())
	else:
	    h = Variable(torch.zeros(test_size, 1, hidden_size))	    
	outputs = []
	#Iterate for length of sequence
	for i in range(sequence_length):
	    if i > 0:
	        _, idx = y.transpose(1, 2).max(1)
	 	y = Variable(idx.data.view(-1, 1).cuda())
	    else:
	        y = output
	    y, h = dec(y, h, i)
	    outputs.append(y.transpose(0, 1)[0])
	output = torch.stack(outputs).transpose(0, 1)

        # get the index of the max log-probability
        _, idx = output[0].data.max(1)
	result_str = [ids[c] for c in idx.squeeze()]
        print("Predicted string: ", ' '.join(result_str))
	print("Actual: " + str(target.data[0][3]))


for epoch in range(1, 10):
    train(epoch)
    test()
