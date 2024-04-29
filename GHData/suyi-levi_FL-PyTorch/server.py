import torch
import torchvision.models as models


class Server(object):
	def __init__(self, conf, eval_dataset):
		self.conf = conf
		self.global_model = models.resnet18().cuda() if torch.cuda.is_available() else models.resnet18()
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

	def model_aggregate(self, weight_accumulator):
		"""
		Aggregate model weights of all selected clients according to the FedAvg algorithm.

		:param weight_accumulator: The accumulated difference between weights of all selected clients.

		"""
		for name, data in self.global_model.state_dict().items():
			updata_per_layer = weight_accumulator[name] * self.conf["lambda"]
			if data.type() != updata_per_layer.type():
				data.add_(updata_per_layer.to(torch.int64))
			else:
				data.add_(updata_per_layer)

	def model_eval(self):
		"""
		Evaluate the performance of the current global model using the evaluation dataset.
		"""
		self.global_model.eval()
		loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			dataset_size += data.size()[0]
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			output = self.global_model(data)
			loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
			pred = output.max(1)[1]
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
		acc = 100 * (float(correct) / float(dataset_size))
		total_loss = loss / dataset_size
		return acc, total_loss