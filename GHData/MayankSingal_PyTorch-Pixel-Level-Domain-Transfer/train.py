import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import data
import model
import numpy as np


def train_val():

	generator = model.Generator().cuda() # Generator
	discriminatorD = model.DiscriminatorD().cuda() # Real-Fake Discriminator
	discriminatorA = model.DiscriminatorA().cuda() # Domain Discriminator

	dataFeeder = data.domainTransferLoader('/home/user/data/lookbook/data')
	train_loader = torch.utils.data.DataLoader(dataFeeder, batch_size=128, shuffle=True,
											   num_workers=2, pin_memory=True)

	criterion = nn.BCEWithLogitsLoss().cuda()

	optimizerD = torch.optim.Adam(discriminatorD.parameters(), lr=0.0002)
	optimizerA = torch.optim.Adam(discriminatorA.parameters(), lr=0.0002)
	optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0002)

	generator.train()
	discriminatorD.train()
	discriminatorA.train()
	
	for epoch in range(10):
		for i, (image1, image2, image3) in enumerate(train_loader):

			I1_var = image1.to(torch.float32).cuda() #Image of cloth being worn by model in image3
			I2_var = image2.to(torch.float32).cuda() #Image of cloth unassociated with model in image3
			I3_var = image3.to(torch.float32).cuda() #Image of Model
			
			real_label_var = torch.ones((I1_var.shape[0],1), requires_grad=False).cuda()
			fake_label_var = torch.zeros((I1_var.shape[0],1), requires_grad=False).cuda()			

			# ----------
			# Train DiscriminatorD
			# ----------
			
			optimizerD.zero_grad()

			out_associated = discriminatorD(I1_var)
			lossD_real_1 = criterion(out_associated, real_label_var)

			out_not_associated = discriminatorD(I2_var)
			lossD_real_2 = criterion(out_not_associated, real_label_var)

			fake = generator(I3_var).detach()
			out_fake = discriminatorD(fake)
			lossD_fake = criterion(out_fake, fake_label_var)

			lossD = (lossD_real_1 + lossD_real_2 + lossD_fake)/3

			lossD.backward()
			optimizerD.step()
			
			# ----------
			# Train DiscriminatorA
			# ----------
			
			optimizerA.zero_grad()

			associated_pair_var = torch.cat((I3_var, I1_var),1)
			not_associated_pair_var = torch.cat((I3_var, I2_var),1)

			fake = generator(I3_var).detach()
			fake_pair_var = torch.cat((I3_var, fake),1)

			out_associated = discriminatorA(associated_pair_var)
			lossA_ass = criterion(out_associated, real_label_var)

			out_not_associated = discriminatorA(not_associated_pair_var)
			lossA_not_ass = criterion(out_not_associated, fake_label_var)

			out_fake = discriminatorA(fake_pair_var)
			lossA_fake = criterion(out_fake, fake_label_var)

			lossA = (lossA_ass + lossA_not_ass + lossA_fake)/3

			lossA.backward()
			optimizerA.step()

			# ----------
			# Train Generator
			# ----------
			
			optimizerG.zero_grad()
			
			fake = generator(I3_var)
			outputD = discriminatorD(fake)
			lossGD = criterion(outputD,real_label_var)

			fake_pair_var = torch.cat((I3_var, fake),1)
			outputA = discriminatorA(fake_pair_var)
			lossGA = criterion(outputA,real_label_var)

			lossG = (lossGD + lossGA)/2

			lossG.backward()
			optimizerG.step()

			if((i+1) % 10) == 0:
				print("Iter:", i+1, "/", len(train_loader))
				print("LossG:", lossG.item(), "LossD:", lossD.item(), "LossA:", lossA.item())
			if((i+1) % 100) == 0:
				torchvision.utils.save_image((fake+1)/2, 'samples/'+str(i+1)+'.jpg')




if __name__ == '__main__':
	os.system('mkdir -p samples')
	train_val()















