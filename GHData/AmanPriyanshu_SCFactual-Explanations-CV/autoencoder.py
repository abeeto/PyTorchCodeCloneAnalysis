import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

class MNIST_AEModel(torch.nn.Module):
	def __init__(self):
		super(MNIST_AEModel, self).__init__()
		self.encoder = torch.nn.Sequential(
											torch.nn.Linear(1*28*28, 526),
											torch.nn.ReLU(),
											torch.nn.Linear(526, 128),
											torch.nn.Sigmoid(),
			)
		self.decoder = torch.nn.Sequential(
											torch.nn.Linear(128, 526),
											torch.nn.ReLU(),
											torch.nn.Linear(526, 1*28*28),
											torch.nn.Sigmoid()
			)
		self.flatten_layer = torch.nn.Flatten()

	def forward(self, x):
		og_shape = x.shape
		x = self.flatten_layer(x)
		x = self.encoder(x)
		x = self.decoder(x)
		return x.view(og_shape)

class ConstructClassifier:
	def __init__(self, dataset, dataset_val):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.x, self.y = dataset.T[1:].T, dataset.T[0]
		self.x = np.reshape(self.x, (len(dataset), 1, 28, 28))/255
		self.x_val, self.y_val = dataset_val.T[1:].T, dataset_val.T[0]
		self.x_val = np.reshape(self.x_val, (len(dataset_val), 1, 28, 28))/255
		self.model = MNIST_AEModel()
		self.model.to(self.device)
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

	def train_single_epoch(self, epoch=0, batch_size=8):
		bar = tqdm(range(0, len(self.y), batch_size))
		running_loss = 0.0
		self.model.train()
		for batch_idx, i in enumerate(bar):
			batch_x = torch.from_numpy(self.x[i:i+batch_size])
			batch_x = batch_x.to(self.device)
			self.optimizer.zero_grad()
			out = self.model(batch_x.float())
			loss = self.criterion(out, batch_x.float())
			loss.backward()
			self.optimizer.step()
			running_loss+=loss.item()
			bar.set_description(str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 3)}))
		bar.close()
		running_loss_val = 0.0
		self.model.eval()
		for batch_idx_val, i in enumerate(range(0, len(self.y_val), batch_size)):
			batch_x = torch.from_numpy(self.x_val[i:i+batch_size])
			batch_x = batch_x.to(self.device)
			out = self.model(batch_x.float())
			loss = self.criterion(out, batch_x.float())
			running_loss_val+=loss.item()
		print("Training Performance:", {"loss": round(running_loss/(batch_idx+1), 3)})
		print("Validation Performance:", {"loss": round(running_loss_val/(batch_idx_val+1), 3)})
		return running_loss/(batch_idx+1), running_loss_val/(batch_idx+1)

	def train(self, epochs):
		for epoch in range(epochs):
			self.train_single_epoch(epoch)
		return self.model

	def predict(self, image):
		self.model.eval()
		image = torch.from_numpy(np.reshape(image, (1, 1, 28, 28))).float().to(self.device)
		return self.model(image).cpu().detach().numpy()[0][0]

if __name__ == '__main__':
	df = pd.read_csv("./data/mnist_train.csv")
	df = df.values
	df_val = pd.read_csv("./data/mnist_val.csv")
	df_val = df_val.values
	cc = ConstructClassifier(df, df_val)
	if not os.path.exists("./output/ae_model.pt"):
		model = cc.train(50)
		torch.save(model, "./output/ae_model.pt")
	else:
		model = torch.load("./output/ae_model.pt")
		cc.model = model
	image = df_val[0][1:]
	image = np.reshape(image, (28, 28))/255
	image_gen = cc.predict(image)
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(image)
	ax1.set_title("Original Image")
	ax2.imshow(image_gen)
	ax2.set_title("AutoEncoder Generated")
	plt.savefig("./images/ae.png")

