import torch
import time, re, os
from options import Options

def save_model(model, epoch, options:Options):
	if not os.path.exists(options.args.model_path):
		os.mkdir(options.args.model_path)

	torch.save(model.state_dict(), "%s/epoch_%d_%d.model" % (
			options.args.model_path,
			epoch,
			int(time.time())
		))

def find_model_file(path):
	max_epoch,max_time = 0,0
	model_name = None
	reg = re.compile(r'epoch_(\d+)_(\d+).model')
	for name in os.listdir(path):
		match = reg.match(name)
		if match:
			time = int(match.group(2))
			epoch = int(match.group(1))
			if time > max_time or (time == max_time and epoch >max_epoch):
				max_time = time
				max_epoch = epoch
				model_name = name

	return model_name

def load_model(model, options:Options):
	path = options.args.model_load
	if path == 'latest':
		if os.path.exists(options.args.model_path):
			path = find_model_file(options.args.model_path)

	if path is None or not os.path.exists(f'{options.args.model_path}/{path}'):
		raise ValueError("Can't find path, ensure you type the true path")
	else:
		path = f'{options.args.model_path}/{path}'
		print(f"load model from: {path}")
		model.load_state_dict(torch.load(path))