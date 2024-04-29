import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

class MNISTModel(torch.nn.Module):
	def __init__(self):
		super(MNISTModel, self).__init__()
		self.cnn = torch.nn.Conv2d(1, 4, kernel_size=3)
		self.linear_1 = torch.nn.Linear(2704, 128)
		self.output_linear = torch.nn.Linear(128, 10)
		self.activation = torch.nn.ReLU()
		self.flatten_layer = torch.nn.Flatten()

	def forward(self, x):
		x = self.cnn(x)
		x = self.flatten_layer(x)
		x = self.linear_1(x)
		x = self.activation(x)
		x = self.output_linear(x)
		return x

class ConstructClassifier:
	def __init__(self, dataset, dataset_val):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.x, self.y = dataset.T[1:].T, dataset.T[0]
		self.x = np.reshape(self.x, (len(dataset), 1, 28, 28))/255
		self.x_val, self.y_val = dataset_val.T[1:].T, dataset_val.T[0]
		self.x_val = np.reshape(self.x_val, (len(dataset_val), 1, 28, 28))/255
		self.model = MNISTModel()
		self.model.to(self.device)
		self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

	def train_single_epoch(self, epoch=0, batch_size=8):
		bar = tqdm(range(0, len(self.y), batch_size))
		running_loss = 0.0
		running_acc = 0.0
		self.model.train()
		for batch_idx, i in enumerate(bar):
			batch_x, batch_y = torch.from_numpy(self.x[i:i+batch_size]), torch.from_numpy(self.y[i:i+batch_size])
			batch_x = batch_x.to(self.device)
			batch_y = batch_y.to(self.device)
			self.optimizer.zero_grad()
			out = self.model(batch_x.float())
			loss = self.criterion(out, batch_y)
			loss.backward()
			self.optimizer.step()
			pred = torch.argmax(out, dim=1)
			acc = torch.mean((pred==batch_y).float())
			running_acc+=acc.item()
			running_loss+=loss.item()
			bar.set_description(str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 3), "acc": round(running_acc/(batch_idx+1), 3)}))
		bar.close()
		running_loss_val = 0.0
		running_acc_val = 0.0
		self.model.eval()
		for batch_idx_val, i in enumerate(range(0, len(self.y_val), batch_size)):
			batch_x, batch_y = torch.from_numpy(self.x_val[i:i+batch_size]), torch.from_numpy(self.y_val[i:i+batch_size])
			batch_x = batch_x.to(self.device)
			batch_y = batch_y.to(self.device)
			out = self.model(batch_x.float())
			loss = self.criterion(out, batch_y)
			pred = torch.argmax(out, dim=1)
			acc = torch.mean((pred==batch_y).float())
			running_acc_val+=acc.item()
			running_loss_val+=loss.item()
		print("Training Performance:", {"loss": round(running_loss/(batch_idx+1), 3), "acc": round(running_acc/(batch_idx+1), 3)})
		print("Validation Performance:", {"loss": round(running_loss_val/(batch_idx_val+1), 3), "acc": round(running_acc_val/(batch_idx_val+1), 3)})
		return running_loss/(batch_idx+1), running_acc/(batch_idx+1), running_loss_val/(batch_idx+1), running_acc_val/(batch_idx+1)

	def train(self, epochs):
		for epoch in range(epochs):
			self.train_single_epoch(epoch)
		return self.model

if __name__ == '__main__':
	df = pd.read_csv("./data/mnist_train.csv")
	df = df.values
	df_val = pd.read_csv("./data/mnist_val.csv")
	df_val = df_val.values
	cc = ConstructClassifier(df, df_val)
	model = cc.train(3)
	torch.save(model, "./output/model.pt")