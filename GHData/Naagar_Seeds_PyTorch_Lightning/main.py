import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy

from load_data import seeds_dataset                         # load the data Set

from torch.utils.data import Dataset, DataLoader 			# Data loader 


import torchvision.models as models							# Model load



# Hyper parameters 

input_size = 128*128*3

num_classes = 4
learning_rate = 0.001
batch_size  = 256
num_epochs = 600


# dataset paths 
train_txt_path='data/train_data_file.csv' 
train_img_dir='data/train'
test_text_path='data/test_data_file.csv'
test_img_dir='data/validation'

# Load dataset. train and test
train_dataset = seeds_dataset(train_txt_path,train_img_dir)
test_dataset = seeds_dataset(test_text_path,test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
validation_loader = DataLoader(test_dataset, batch_size=batch_size)

### Selecting model for training 

vgg16 = models.vgg16()
resnet18 = models.resnet18()
resnet34 = models.resnet34()
squeezenet_1 = models.squeezenet1_0()
squeezenet_2 = models.squeezenet1_1()

# model = vgg16
# model = resnet18
# model = resnet34
# model = squeezenet_1
# model = squeezenet_2
# model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True) ## Squeezenet_Pretrained 
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True) ## Alexnet_pretrained
# model = SqueezeNet()
# model  = seeds_model()
# model = MobileNetV2(width_mult=1)
# model = ResNet18()

# model = ResNet50(img_channel=3, num_classes=4)
# model = ResNet101(img_channel=3,num_classes=4)
# model = ResNet152(img_channel=3, num_classes=4)

class Lit_NN(pl.LightningModule):
	def __init__(self, num_classes, model):
		super(Lit_NN, self).__init__()
		self.alexnet = model 

	def forward(self, x):
		out = self.alexnet(x)

		return out

	def training_step(self, batch, batch_idx):

		images, labels = batch
		# images = images.reshape(-1, 3*128*128)


		# Forward Pass
		outputs = self(images)
		loss = F.cross_entropy(outputs, labels)
		# acc_i = self.loss(outputs, labels)
		acc = accuracy(outputs, labels)
		pbar = {'train_acc': acc}

		tensorboard_logs = {'train_loss': loss, 'train_acc': pbar}
		return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': pbar}
	

	def configure_optimizers(self):

		return torch.optim.Adam(model.parameters(), lr=learning_rate)

	def train_dataloader(self):

		
		train_dataset = seeds_dataset(train_txt_path,train_img_dir)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)


		return train_loader
	def val_dataloader(self):

		

		test_dataset = seeds_dataset(test_text_path,test_img_dir)
		val_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)


		return val_loader

	def validation_step(self, batch, batch_idx):

		images, labels = batch
		


		# Forward Pass
		outputs = self(images)
		val_loss = F.cross_entropy(outputs, labels)

		# acc_val = self.val_loss(outputs, labels)
		acc = accuracy(outputs, labels)
		tensorboard_logs = {'val_loss': val_loss, 'val_acc': acc}
		
		
		return {'val_loss': val_loss, 'log': tensorboard_logs, 'val_acc':acc}

	def validation_epoch_ends(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
		pbar = {'val_acc': acc}

		tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

		return {'val_loss': avg_loss, 'log': tensorboard_logs,'progress_bar': pbar}


if __name__ == '__main__':

	trainer = Trainer(auto_lr_find=True, max_epochs=num_epochs, fast_dev_run=False,gpus=1) # 'fast_dev_run' for checking errors, "auto_lr_find" to find the best lr_rate
	model = Lit_NN(num_classes, model)

	trainer.fit(model)






####### ------------------- TRASH ---------------------######## 


# self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)