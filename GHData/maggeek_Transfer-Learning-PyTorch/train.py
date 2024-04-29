import math
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.autograd import Variable
import utils
from config import args
from sklearn.metrics import confusion_matrix
import model


if __name__ == '__main__':

	cuda = True if torch.cuda.is_available() else False
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	print(cuda)

	# load dataset
	trainloader, validloader, class_names = utils.load_dataset()

	# load or create model
	if args.pretrained:
		# use pretrained model
		print("=> using pre-trained model '{}'".format(args.arch))
			# Freeze model weights
		model = models.__dict__[args.arch]()
		for param in model.parameters():
			param.requires_grad = False
		# overwrite the last layer with correct number of classes
		num_ftrs = model.fc.in_features
		print(num_ftrs)
		#model.classifier[6] = nn.Linear(num_ftrs, num_classes)
		model.fc = nn.Linear(num_ftrs, len(class_names))
		# Add on classifier
		#model.fc = nn.Sequential(
		#                      nn.Linear(num_ftrs, 256),
		#                      nn.ReLU(),
		#                      nn.Dropout(0.4),
		#                      nn.Linear(256, num_classes))
		# Only training classifier[6]
		print(model.fc)
	else:
		# train model from scratch
		#print("=> creating model '{}'".format(args.arch))
		#model = models.__dict__[args.arch](pretrained=args.pretrained)
		# load model
		model = model.CNN()
	print(model)

	#valid = [validset[i] for i in range(3)][0]
	#valid = np.asarray(valid).type(Tensor)

	# Find total parameters and trainable parameters
	total_params = sum(p.numel() for p in model.parameters())
	print(f'{total_params:,} total parameters.')
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'{total_trainable_params:,} training parameters.')

	# choose loss function​
	criterion = nn.CrossEntropyLoss()
	#optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
	#optimizer = torch.optim.SGD(model.parameters(),
	#							lr=args.lr,
	#                            momentum=args.momentum,
	#                            weight_decay=args.weight_decay)
	# choose optimizer​
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
	if cuda:
		criterion.cuda()
		model.cuda()
	
	total_train_loss, total_train_acc, total_valid_loss, total_valid_acc = [], [], [], []
	for epoch in range(args.n_epochs):

		n_train_correct, n_train_total, n_valid_correct, n_valid_total = 0, 0, 0, 0
		model.train()
		for batch, (images, gtruth) in enumerate(trainloader):

			input_var = Variable(images.type(Tensor))
			gtruth_var = Variable(gtruth.type(Tensor))

			# compute output
			output = model(input_var)
			gt = gtruth_var.long()
			loss = criterion(output, gt)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# calculate accuracy of predictions in the current batch
			n_train_correct += (torch.max(output, 1)[1].view(gt.size()) == gt).sum().item()
			n_train_total += args.batch_size
			train_acc = 100. * n_train_correct / n_train_total

			# save loss and accuracy
			utils.save_loss(epoch+1, batch+1, len(trainloader), loss.item(), train_acc, 'training')

		total_train_loss.append(loss.item())
		total_train_acc.append(train_acc)

		predicted = []
		real = []
		# img size to arguments, check how it depends on dataset
		valid = torch.zeros([100, args.n_channels, 224, 224])
		model.eval()
		for batch, (images, gtruth) in enumerate(validloader):

			input_var = Variable(images.type(Tensor))
			gtruth_var = Variable(gtruth.type(Tensor))

			# will not work if batch < 10, calculate how many batches need to be taken instead
			if batch < 10:
				valid[batch*10:batch*10+10, :, :, :] = images[0:10, :, :, :].type(Tensor)
			
			# compute output
			output = model(input_var)
			gt = gtruth_var.long()
			loss = criterion(output, gt)
			real.extend(gtruth.numpy())
			predicted.extend(torch.max(output.detach(), 1)[1].cpu().numpy())

			# calculate accuracy of predictions in the current batch
			n_valid_correct += (torch.max(output, 1)[1].view(gt.size()) == gt).sum().item()
			n_valid_total += args.batch_size
			valid_acc = 100. * n_valid_correct / n_valid_total

			# save loss and accuracy
			utils.save_loss(epoch+1, batch+1, len(validloader), loss.item(), valid_acc, 'validation')

		# save confusion matrix
		cm = confusion_matrix(real, predicted)
		utils.save_cm(cm, class_names, epoch)
		print('Confusion matrix saved')

		total_valid_loss.append(loss.item())
		total_valid_acc.append(valid_acc)

		#if epoch % 5 == 0:
		utils.save_plot(epoch+1, total_train_loss, 'Training loss')
		utils.save_plot(epoch+1, total_train_acc, 'Training accuracy')
		utils.save_plot(epoch+1, total_valid_loss, 'Validation loss')
		utils.save_plot(epoch+1, total_valid_acc, 'Validation accuracy')
		#torch.save(model.state_dict(), '{}/encoder.pth'.format(opt.folder_name))
		print('Plots saved')
		n_images = 100
		utils.save_images(class_names, valid, predicted[0:n_images], epoch, int(math.sqrt(n_images)))
		print('Images saved')
