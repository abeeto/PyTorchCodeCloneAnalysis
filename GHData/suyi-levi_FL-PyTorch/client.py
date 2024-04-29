import torch


class Client(object):
	def __init__(self, conf, model, train_dataset, id=1):
		self.conf = conf
		self.local_model = model
		self.id = id
		self.train_dataset = train_dataset
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset))
		indices = all_range[id * data_len: (id + 1) * data_len]
		self.train_loader = torch.utils.data.DataLoader(
			self.train_dataset,
			batch_size=conf["batch_size"], 
			sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

	def local_train(self, model):
		"""
		Train local model of clients.

		:param model: The global model downloaded from the server.
		"""
		# Copy parameters from the global model to the local model
		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())

		# Define optimizer for local training
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])

		# Train local model
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
				optimizer.step()
			print("Epoch %d done." % e)
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
		return diff
