import logging
import logging
import os
import random
import shutil
import warnings
from collections import OrderedDict
from copy import deepcopy, copy
from typing import Any, Dict, Iterable

import pandas as pd
import psutil
import torch
import tqdm
import neurotorch as nt
from neurotorch import Dimension, DimensionProperty
from neurotorch.callbacks import CheckpointManager, LoadCheckpointMode
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.early_stopping import EarlyStoppingThreshold
from neurotorch.callbacks.training_visualization import TrainingHistoryVisualizationCallback
from neurotorch.metrics import ClassificationMetrics
from neurotorch.modules import SequentialModel, SpikeFuncType
from neurotorch.modules.layers import LayerType, LayerType2Layer, LearningType
from neurotorch.trainers import ClassificationTrainer
from neurotorch.utils import get_all_params_combinations, hash_params, save_params, set_seed
from utils import get_transform_from_str

from dataset import DatasetId, get_dataloaders


def get_training_params_space() -> Dict[str, Any]:
	"""
	Get the parameters space for the training.
	:return: The parameters space.
	"""
	return {
		"dataset_id"              : [
			DatasetId.MNIST,
			DatasetId.FASHION_MNIST
		],
		"input_transform"         : [
			# "linear",
			"NorseConstCurrLIF",
			# "ImgToSpikes",
			"const",
			"spylif",
			"spyalif",
			"alif",
			"lif",
		],
		"n_steps"                 : [
			8,
			16,
			32,
			64,
			100,
			1_000
		],
		"n_hidden_neurons"        : [
			16,
			32,
			64,
			[64, 64],
			128,
			256,
			[32, 32],
			32
		],
		"spike_func"              : [SpikeFuncType.FastSigmoid, ],
		"hidden_layer_type"       : [
			LayerType.LIF,
			LayerType.ALIF,
			LayerType.SpyLIF,
		],
		"readout_layer_type"      : [
			LayerType.LI,
			LayerType.SpyLI,
		],
		"use_recurrent_connection": [
			False,
			True
		],
		"learn_beta"              : [
			True,
			False
		],
	}


def set_default_params(params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
	params.setdefault("input_transform", "spylif")
	params.setdefault("n_steps", 8)
	params.setdefault("n_hidden_neurons", 128)
	params.setdefault("hidden_layer_type", nt.LayerType.LIF)
	params.setdefault("readout_layer_type", nt.LayerType.SpyLI)
	params.setdefault("use_recurrent_connection", False)
	params.setdefault("learn_beta", False)
	params.setdefault("foresight_time_steps", 0)
	params.setdefault("seed", kwargs["seed"])
	return params


def train_with_params(
		params: Dict[str, Any],
		n_iterations: int = 100,
		batch_size: int = 256,
		data_folder: str = "tr_results",
		verbose: bool = False,
		show_training: bool = False,
		force_overwrite: bool = False,
		seed: int = 0,
		nb_workers=0,
		pin_memory: bool = True,
):
	params = set_default_params(params, seed=seed)
	set_seed(seed)
	checkpoints_name = str(hash_params(params))
	checkpoint_folder = f"{data_folder}/{checkpoints_name}"
	os.makedirs(checkpoint_folder, exist_ok=True)
	if verbose:
		logging.info(f"Checkpoint folder: {checkpoint_folder}")
	
	n_features = 28 * 28
	
	dataloaders = get_dataloaders(
		dataset_id=params["dataset_id"],
		batch_size=batch_size,
		train_val_split_ratio=params.get("train_val_split_ratio", 0.85),
		nb_workers=nb_workers,
		pin_memory=pin_memory,
	)
	n_hidden_neurons = params["n_hidden_neurons"]
	if not isinstance(n_hidden_neurons, Iterable):
		n_hidden_neurons = [n_hidden_neurons]
	n_hidden_neurons.insert(0, n_features)
	
	hidden_layers = [
		LayerType2Layer[params["hidden_layer_type"]](
			input_size=n_hidden_neurons[i],
			output_size=n,
			# spike_func=SpikeFuncType2Func[params["spike_func"]],
			**params
		)
		for i, n in enumerate(n_hidden_neurons[1:])
	]
	input_params = deepcopy(params)
	input_params.pop("forward_weights", None)
	input_params.pop("use_recurrent_connection", None)
	input_params.pop("learning_type", None)
	network = SequentialModel(
		input_transform=get_transform_from_str(params["input_transform"], **params, n_units=n_features),
		layers=[
			*hidden_layers,
			LayerType2Layer[params["readout_layer_type"]](output_size=10),
		],
		name=f"{params['dataset_id'].name}_network_{checkpoints_name}",
		checkpoint_folder=checkpoint_folder,
		foresight_time_steps=params["foresight_time_steps"],
	).build()
	if verbose:
		logging.info(f"\nNetwork:\n{network}")
	checkpoint_manager = CheckpointManager(checkpoint_folder, metric="val_accuracy", minimise_metric=False)
	save_params(params, os.path.join(checkpoint_folder, "params.pkl"))
	convergence_time_getter = ConvergenceTimeGetter(metric='val_accuracy', threshold=0.95, minimize_metric=False)
	callbacks = [
		checkpoint_manager,
		convergence_time_getter,
		EarlyStoppingThreshold(metric='val_accuracy', threshold=0.99, minimize_metric=False),
	]
	if show_training:
		callbacks.append(TrainingHistoryVisualizationCallback("./temp/"))
	trainer = ClassificationTrainer(
		model=network,
		callbacks=callbacks,
		verbose=verbose,
	)
	history = trainer.train(
		dataloaders["train"],
		dataloaders["val"],
		n_iterations=n_iterations,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR if not force_overwrite else None,
		force_overwrite=force_overwrite,
		exec_metrics_on_train=False,
	)
	try:
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.BEST_ITR, verbose=verbose)
	except FileNotFoundError:
		if verbose:
			logging.info("No best checkpoint found. Loading last checkpoint instead.")
		network.load_checkpoint(checkpoint_manager.checkpoints_meta_path, LoadCheckpointMode.LAST_ITR, verbose=verbose)
	
	predictions = {
		k: ClassificationMetrics.compute_y_true_y_pred(network, dataloader, verbose=verbose, desc=f"{k} predictions")
		for k, dataloader in dataloaders.items()
	}
	return OrderedDict(
		dict(
			params=params,
			network=network,
			checkpoints_name=checkpoints_name,
			history=history,
			convergence_time_getter=convergence_time_getter,
			accuracies={
				k: ClassificationMetrics.accuracy(network, y_true=y_true, y_pred=y_pred)
				for k, (y_true, y_pred) in predictions.items()
			},
			precisions={
				k: ClassificationMetrics.precision(network, y_true=y_true, y_pred=y_pred)
				for k, (y_true, y_pred) in predictions.items()
			},
			recalls={
				k: ClassificationMetrics.recall(network, y_true=y_true, y_pred=y_pred)
				for k, (y_true, y_pred) in predictions.items()
			},
			f1s={
				k: ClassificationMetrics.f1(network, y_true=y_true, y_pred=y_pred)
				for k, (y_true, y_pred) in predictions.items()
			},
		)
	)


def train_all_params(
		training_params: Dict[str, Any] = None,
		rm_data_folder_and_restart_all_training: bool = False,
		skip_if_exists: bool = False,
		shuffle_list_of_params: bool = True,
		**train_with_params_kwargs,
):
	"""
	Train the network with all the parameters.

	:param shuffle_list_of_params: Shuffle the list of parameters before training.
	:param training_params: The parameters to use for the training.
	:param rm_data_folder_and_restart_all_training: If True, remove the data folder and restart all the training.
	:param skip_if_exists: If True, skip the training if the results already in the results dataframe.

	:keyword str data_folder: The folder where to save the data.

	:return: The results of the training.
	"""
	warnings.filterwarnings("ignore", category=UserWarning)
	train_with_params_kwargs.setdefault("data_folder", "tr_results")
	data_folder = train_with_params_kwargs["data_folder"]
	if rm_data_folder_and_restart_all_training and os.path.exists(data_folder):
		shutil.rmtree(data_folder)
	os.makedirs(data_folder, exist_ok=True)
	results_path = os.path.join(data_folder, "results.csv")
	if training_params is None:
		training_params = get_training_params_space()
	
	all_params_combinaison_dict = get_all_params_combinations(training_params)
	if shuffle_list_of_params:
		random.shuffle(all_params_combinaison_dict)
	columns = [
		'checkpoints',
		*list(training_params.keys()),
		'train_accuracy', 'val_accuracy', 'test_accuracy',
		'train_precision', 'val_precision', 'test_precision',
		'train_recall', 'val_recall', 'test_recall',
		'train_f1', 'val_f1', 'test_f1',
	]
	
	# load dataframe if exists
	try:
		df = pd.read_csv(results_path)
	except FileNotFoundError:
		df = pd.DataFrame(columns=columns)
	logging.info(f"Training {len(all_params_combinaison_dict)} networks. {len(df)} already trained.")
	with tqdm.tqdm(all_params_combinaison_dict, desc="Training all the parameters", position=0) as p_bar:
		for i, params in enumerate(p_bar):
			if str(hash_params(params)) in df["checkpoints"].values and skip_if_exists:
				continue
			# p_bar.set_description(f"Training {params}")
			try:
				df = _do_iteration_of_all_params(
					df, p_bar, results_path,
					params,
					**train_with_params_kwargs
				)
			except RuntimeError as e:
				smaller_params = copy(train_with_params_kwargs)
				smaller_params["batch_size"] = 128
				df = _do_iteration_of_all_params(
					df, p_bar, results_path,
					params,
					**smaller_params
				)
			except Exception as e:
				logging.error(e)
				continue
	
	logging.info(f"Training done. {len(df)} networks trained. Data saved in {results_path}:")
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		logging.info(df)
	return df


def _do_iteration_of_all_params(
		df, p_bar, results_path,
		params,
		**train_with_params_kwargs
):
	result = train_with_params(
		params,
		show_training=False,
		**train_with_params_kwargs
	)
	params.update(result["params"])
	convergence_time_getter: ConvergenceTimeGetter = result["convergence_time_getter"]
	training_time = convergence_time_getter.training_time
	itr_convergence = convergence_time_getter.itr_convergence
	time_convergence = convergence_time_getter.time_convergence
	convergence_thr = convergence_time_getter.threshold
	if result["checkpoints_name"] in df["checkpoints"].values:
		training_time = df.loc[df["checkpoints"] == result["checkpoints_name"], "training_time"].values[0]
		itr_convergence = df.loc[df["checkpoints"] == result["checkpoints_name"], "itr_convergence"].values[0]
		time_convergence = df.loc[df["checkpoints"] == result["checkpoints_name"], "time_convergence"].values[0]
		convergence_thr = df.loc[df["checkpoints"] == result["checkpoints_name"], "convergence_thr"].values[0]
		# remove from df if already exists
		df = df[df["checkpoints"] != result["checkpoints_name"]]
	df = pd.concat(
		[df, pd.DataFrame(
			dict(
				checkpoints=[result["checkpoints_name"]],
				**{k: [v] for k, v in params.items()},
				training_time=training_time,
				itr_convergence=itr_convergence,
				time_convergence=time_convergence,
				convergence_thr=convergence_thr,
				# accuracies
				train_accuracy=[result["accuracies"]["train"]],
				val_accuracy=[result["accuracies"]["val"]],
				test_accuracy=[result["accuracies"]["test"]],
				# precisions
				train_precision=[result["precisions"]["train"]],
				val_precision=[result["precisions"]["val"]],
				test_precision=[result["precisions"]["test"]],
				# recalls
				train_recall=[result["recalls"]["train"]],
				val_recall=[result["recalls"]["val"]],
				test_recall=[result["recalls"]["test"]],
				# f1s
				train_f1=[result["f1s"]["train"]],
				val_f1=[result["f1s"]["val"]],
				test_f1=[result["f1s"]["test"]],
			)
		)], ignore_index=True,
	)
	df.to_csv(results_path, index=False)
	p_bar.set_postfix(
		params=params,
		train_accuracy=result["accuracies"]['train'],
		val_accuracy=result["accuracies"]['val'],
		test_accuracy=result["accuracies"]['test']
	)
	return df
