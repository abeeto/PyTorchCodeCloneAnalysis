import pprint

import torch
from pythonbasictools.device import DeepLib, log_device_setup
from pythonbasictools.logging import logs_file_setup

from dataset import DatasetId
from results_generation import train_with_params
from neurotorch.modules.layers import LayerType, LearningType

if __name__ == '__main__':
	logs_file_setup(__file__, add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)
	if torch.cuda.is_available():
		torch.cuda.set_per_process_memory_fraction(0.8)
	results = train_with_params(
		{
			"dataset_id": DatasetId.MNIST,
			"use_recurrent_connection": False,
			"input_transform": "spyalif",
			'n_hidden_neurons': 128,
			"n_steps": 8,
			"train_val_split_ratio": 0.95,
			# "spike_func": SpikeFuncType.FastSigmoid,
			"hidden_layer_type": LayerType.LIF,
			"readout_layer_type": LayerType.SpyLI,
		},
		n_iterations=30,
		batch_size=1024,
		verbose=True,
		show_training=False,
		force_overwrite=False,
		data_folder="data/tr_test",
		nb_workers=2,
		pin_memory=True,
	)
	pprint.pprint(results, indent=4)
	results["history"].plot(show=True)
