import numpy as np
import neurotorch as nt
from neurotorch.transforms.spikes_encoders import SpyLIFEncoder, LIFEncoder, ALIFEncoder, SpikesEncoder


def get_transform_from_str(transform_name: str, **kwargs):
	"""
	Get a transform from a string. The string should be one of the following:
	- "none": No transform.
	- "linear": Linear transform.
	- "ImgToSpikes": Image to spikes transform.
	- "NorseConstCurrLIF": Norse constant current LIF transform.
	- "flatten": Flatten transform.
	- "constant": Constant transform.

	:param transform_name: The name of the transform.
	:param kwargs: The arguments for the transform.

	:keyword Arguments:
		* <dt>: float -> The time step of the transform.
		* <n_steps>: float -> The number of times steps of the transform.
		* <n_units>: int -> The number of units of the transform.

	:return: The transform.
	"""
	import torch
	from torchvision.transforms import Compose
	from neurotorch.transforms import LinearRateToSpikes
	import norse
	from neurotorch.transforms.vision import ImgToSpikes
	from torchvision.transforms import Lambda
	from neurotorch.transforms import ConstantValuesTransform
	
	kwargs.setdefault("dt", 1e-3)
	kwargs.setdefault("n_steps", 10)
	kwargs.setdefault("n_units", 28*28)
	
	name_to_transform = {
		"none"             : None,
		"linear"           : Compose([torch.nn.Flatten(start_dim=2), LinearRateToSpikes(n_steps=kwargs["n_steps"])]),
		"NorseConstCurrLIF": Compose([
				torch.nn.Flatten(start_dim=1),
				norse.torch.ConstantCurrentLIFEncoder(seq_length=kwargs["n_steps"], dt=kwargs["dt"]),
				Lambda(lambda x: x.permute(1, 0, 2))
		]),
		"ImgToSpikes"      : Compose([torch.nn.Flatten(start_dim=2), ImgToSpikes(n_steps=kwargs["n_steps"], use_periods=True)]),
		"flatten"          : torch.nn.Flatten(start_dim=2),
		"const"            : Compose(
			[torch.nn.Flatten(start_dim=2), ConstantValuesTransform(n_steps=kwargs["n_steps"])]
		),
		"spylif"           : Compose(
			[torch.nn.Flatten(start_dim=2), SpyLIFEncoder(n_steps=kwargs["n_steps"], n_units=kwargs["n_units"])]
		),
		"lif": Compose(
			[torch.nn.Flatten(start_dim=2), LIFEncoder(n_steps=kwargs["n_steps"], n_units=kwargs["n_units"])]
		),
		"alif": Compose(
			[torch.nn.Flatten(start_dim=2), ALIFEncoder(n_steps=kwargs["n_steps"], n_units=kwargs["n_units"])]
		),
		"spyalif": Compose([
			torch.nn.Flatten(start_dim=2),
			SpikesEncoder(n_steps=kwargs["n_steps"], n_units=kwargs["n_units"], spikes_layer_type=nt.SpyALIFLayer)
		]),
	}
	name_to_transform = {k.lower(): v for k, v in name_to_transform.items()}
	return name_to_transform[transform_name.lower()]
