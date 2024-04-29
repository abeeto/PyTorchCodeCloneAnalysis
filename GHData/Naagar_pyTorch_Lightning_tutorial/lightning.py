import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl 
from pytorch_lightning import Trainer


# Hyper parameters 

input_size = 784
hidden_size = 500
num_classes = 10
learning_rate = 0.001
batch_size  = 64
num_epochs = 10



# dataset paths 
train_txt_path='seeds_dataset/data/train_data_file.csv' 
train_img_dir='seeds_dataset/data/train'
test_text_path='seeds_dataset/data/test_data_file.csv'
test_img_dir='seeds_dataset/data/validation'

# Load dataset. train and test
train_dataset = seeds_dataset(train_txt_path,train_img_dir)
test_dataset = seeds_dataset(test_text_path,test_img_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
validation_loader = DataLoader(test_dataset, batch_size=batch_size)


class Lit_NN(pl.LightningModule):
	def __init__(self, input_size, hidden_size, num_classes):
		super(Lit_NN, self).__init__()
		self.input_size = input_size
		self.l1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)

		# no activation and no softmax at the end 
		return out


		


	def training_step(self, batch, batch_idx):

		images, labels = batch
		images = images.reshape(-1, 28*28)


		# Forward Pass
		outputs = self(images)
		loss = F.cross_entropy(outputs, labels)
		tensorboard_logs = {'train_loss': loss}
		return {'loss': loss, 'log': tensorboard_logs}
	

	def configure_optimizers(self):

		return torch.optim.Adam(model.parameters(), lr=learning_rate)

	def train_dataloader(self):

		# dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

		# loader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

		train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
		train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size,
													num_workers=4,shuffle=True)


		return train_loader
	def val_dataloader(self):

		# dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

		# loader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

		val_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
		val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size=batch_size,
													num_workers=4)


		return val_loader

	def validation_step(self, batch, batch_idx):

		images, labels = batch
		images = images.reshape(-1, 28*28)


		# Forward Pass
		outputs = self(images)
		val_loss = F.cross_entropy(outputs, labels)
		# tensorboard_logs = {'train_loss:' loss}
		return {'val_loss': val_loss}
	def validation_epoch_ends(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		tensorboard_logs = {'val_loss': avg_loss}

		return {'val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':

	trainer = Trainer(auto_lr_find=True, max_epochs=num_epochs, fast_dev_run=False) # 'fast_dev_run' for checking errors, "auto_lr_find" to find the best lr_rate
	model = Lit_NN(input_size, hidden_size, num_classes)

	trainer.fit(model)