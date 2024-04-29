import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model_classifier import MNISTModel
from matplotlib import pyplot as plt

class CFXai:
	def __init__(self, image, target_y, model_path="./output/model.pt", lr=0.001, lamda=3):
		self.lamda = lamda
		self.x_dash = torch.from_numpy(np.reshape(image, (1, 1, 28, 28))).float()
		self.x = self.x_dash.clone()
		self.target_y = target_y
		self.model = torch.load(model_path)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.x, self.x_dash = self.x.to(self.device), self.x_dash.to(self.device)
		self.ungrad_model()
		self.prob_layer = torch.nn.Softmax(dim=1)
		self.lr = lr
		self.criterion = torch.nn.CrossEntropyLoss()

	def ungrad_model(self):
		for param in self.model.parameters():
			param.requires_grad=False

	def generate_cf_single_iter(self):
		self.x_dash.requires_grad=True
		out = self.model(self.x_dash)
		pred = self.prob_layer(out)
		loss = self.criterion(out, torch.tensor([self.target_y]).to(self.device))
		loss_og = loss.item()
		d = torch.mean(torch.abs(self.x_dash - self.x))
		loss.data = torch.Tensor([d.item()+self.lamda * loss.item()]).to(self.device)[0]
		loss.backward()
		self.x_dash = self.x_dash - self.lr*self.x_dash.grad
		self.x_dash = self.x_dash.detach()
		return loss_og, d.item()

	def generate_cf(self, num_iters):
		bar = tqdm(range(num_iters))
		for epoch in bar:
			loss, d = self.generate_cf_single_iter()
			bar.set_description(str({"epoch": epoch+1, "loss": round(loss, 3), "d": round(d, 3)}))
		bar.close()
		self.x_dash = self.x_dash.cpu().numpy()
		self.x = self.x.cpu().numpy()
		return self.x_dash[0][0], self.x[0][0], loss, d

if __name__ == '__main__':
	df_test = pd.read_csv("./data/mnist_test.csv")
	df_test = df_test.values

	first_image_xy = df_test[10]
	first_image_x = first_image_xy[1:]/255
	first_image_y = first_image_xy[0]
	print(first_image_y)
	cfxai = CFXai(first_image_x, target_y=1, lr=0.01)
	img, og_img, loss, d = cfxai.generate_cf(100)
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(og_img)
	ax1.set_title("Original Image")
	ax2.imshow(img)
	ax2.set_title("Counterfactual")
	plt.savefig("./images/Constructions.png")