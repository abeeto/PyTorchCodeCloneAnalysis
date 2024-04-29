import datasets
from server import Server
from client import Client
import json
import random
import torch


with open('./conf.json', 'r') as f:
	conf = json.load(f)

train_datasets, eval_datasets = datasets.get_dataset('./data/', conf["type"])
server = Server(conf, eval_datasets)
clients = []
for c in range(conf["total_clients"]):
	clients.append(Client(conf, server.global_model, train_datasets, c))

for e in range(conf["global_epochs"]):
	candidates = random.sample(clients, conf["select_clients"])
	weight_accumulator = {}
	for name, params in server.global_model.state_dict().items():
		weight_accumulator[name] = torch.zeros_like(params)

	for c in candidates:
		diff = c.local_train(server.global_model)
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name].add_(diff[name])

	server.model_aggregate(weight_accumulator)  # model aggregation
	acc, loss = server.model_eval()
	print("Epoch % d, acc: %f, loss: %f\n" % (e, acc, loss))