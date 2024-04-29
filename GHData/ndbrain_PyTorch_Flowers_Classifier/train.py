import torch
from model import Model
from torch import optim

import numpy as np

try:
	from ConfigParser import SafeConfigParser
except:
	from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.


class Train(object):
	"""docstring for Train"""
	def __init__(self, config_file='config.ini'):
		super(Train, self).__init__()
		self.config_file = config_file
		self.model = Model()

		self.config = get_config()

		self.epochs = self.config['epochs']
		self.opt = self.config['opt']
		self.lr = self.config['lr']

		self.loss_fn = nn.CrossEntropyLoss()
		self.opt = optim.Adam(model.parameters(), lr)

		cuda = torch.cuda.is_available()

	def get_config(config_file=self.config_file):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

	def run(self, model=self.model, epochs=self.epochs, loss_fn=self.loss_fn, opt=self.opt,):

		valid_loss_min = np.Inf

		if self.cuda:
		  print("Start Training on GPU ......")
		else:
		  print("Start Training on CPU ......")

		for e in range(epochs):
		  train_losses = 0.0
		  valid_losses = 0.0

		  # training
		  for x, y in data_loaders['train']:
		    if cuda:
		      model.cuda()
		      x, y = x.cuda(), y.cuda()
		    model.train()
		    opt.zero_grad()
		    out = model(x)
		    loss = loss_fn(out, y)
		    train_losses += loss.item()*x.size(0)
		    loss.backward()
		    opt.step()
		  
		  # validation
		  with torch.no_grad():
		    class_correct = [0. for i in range(102)]
		    class_total = [0. for i in range(102)]
		    for x, y in data_loaders['valid']:
		      if self.cuda:
		        model.cuda()
		        x, y = x.cuda(), y.cuda()
		      model.eval()
		      out = model(x)
		      loss = loss_fn(out, y)
		      valid_losses += loss.item()*x.size(0)
		      
		      _, p = torch.max(out, 1)
		      correct_tensor = p.eq(y.data.view_as(p))
		      correct = np.squeeze(correct_tensor.numpy() if not cuda else correct_tensor.cpu().numpy())

		      for i in range(x.shape[0]):
		        lab = y.data[i]
		        class_correct[lab] += correct[i].item()
		        class_total[lab] += 1
		  
		  train_loss_avg = train_losses/len(data_loaders['train'].dataset)
		  valid_loss_avg = valid_losses/len(data_loaders['valid'].dataset)
		  
		  print("Epoch: {}/{} ... ".format(e+1, epochs),
		        "Training Loss: {:.6f}".format(train_loss_avg),
		        "Validation Loss: {:.6f}".format(valid_loss_avg),
		        "Accuracy: {:.2f}".format(100 * (np.sum(class_correct) / np.sum(class_total)))
		       )
		  
		  if valid_loss_avg < valid_loss_min:
		    print("\tValidation Loss Decreased {} => {} .... Saving The Model .......".format(valid_loss_min, valid_loss_avg))
		    model.cpu()
		    torch.save(model.state_dict(), 'trained/checkpoint.cpt')
		    valid_loss_min = valid_loss_avg

if __name__ == '__main__':
	train = Train()
	train.run()