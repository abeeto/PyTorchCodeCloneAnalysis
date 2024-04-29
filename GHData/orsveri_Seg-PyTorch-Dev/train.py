import os
import json
import time
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from data.datasets import Dataset_ADE20K
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import deeplabv3_resnet50

from models import save_checkpoint, get_model_constructor


# -------------------------------------------------------------------------------------------------------------------- #
# Parameters and input data
# -------------------------------------------------------------------------------------------------------------------- #

csv_train = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/anno_meta/anno_filelist_train.csv"
csv_test  = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/anno_meta/anno_filelist_val.csv"
dataset_root = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/"
class_file = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/anno_meta/classes_map.csv"

input_shape = (256, 512) # can have the length of 2 (H, W) or 3 (H, W, C)
resize_pad = True # if True, aspect ratio will be kept by using padding
do_aug = False

model_name = "deeplabv3_resnet50"
pretrained_weights = "" # weights with the same number of classes
pretrained_nb_classes = None # None if nb_classes is what we want
logdir = "/media/sveta/DATASTORE/AI_ML_DL/Projects/Segmentation/Seg-Pytorch-Dev_data/logs/try02"
batch_size = 3
epochs = 3
ckpt_every_steps = 10 	# can be None
test_every_steps = 20 	# can be None
restrict_gpu = False	# GPU won't be used if restrict_gpu = True

# Notes for the config file. It may contain anything that can be written to .json file.
notes = "Test run of training framework. Database: ADE20K, default train-test split.\n " \
		"Fixing some small things"

# -------------------------------------------------------------------------------------------------------------------- #
# Prepare the data, directory with logs and config file
# -------------------------------------------------------------------------------------------------------------------- #

classes_df = pd.read_csv(class_file, sep=",")
nb_classes = len(classes_df)

data_train = Dataset_ADE20K(csv_file=csv_train, root_dir=dataset_root, nb_classes=nb_classes, input_shape=input_shape,
							resize_pad=resize_pad)
data_test  = Dataset_ADE20K(csv_file=csv_test, root_dir=dataset_root, nb_classes=nb_classes, input_shape=input_shape,
							resize_pad=resize_pad)

"""
def collate_wrapper(batch):
    return SimpleCustomBatch(batch)
"""
collate_wrapper = None
loader_train = DataLoader(data_train, batch_size=batch_size, collate_fn=collate_wrapper,
					pin_memory=True)
loader_test = DataLoader(data_train, batch_size=batch_size, collate_fn=collate_wrapper,
					pin_memory=True)
L_train = len(loader_train)
L_test  = len(loader_test)

train_config = {}

# Let's check that we won't overwrite existing log files
try:
	os.mkdir(os.path.join(logdir))
except FileExistsError:
	if any(file.startswith('events.out.') for file in os.listdir(os.path.join(logdir))):
		raise Exception(
			"It is restricted to write logs of new training session to a directory with some other log files! \n"
			"Don't mix diferent log files! Please change logdir in script parameters")

train_config["logdir"] = logdir
train_config["classes_file_path"] = class_file
train_config["input_shape"] = [input_shape[0], input_shape[1], 3 if len(input_shape) == 2 else input_shape[2]]
train_config["num_classes"] = nb_classes
train_config["data_train"] = csv_train
train_config["data_valtest"] = csv_test
train_config["resize_pad"] = resize_pad
train_config["with_augment"] = do_aug
train_config["notes"] = notes

# Save config
with open(os.path.join(logdir, "cfg.json"), 'w', encoding="utf-8") as f:
	json.dump(train_config, f, indent=4)

# -------------------------------------------------------------------------------------------------------------------- #
# Prepare the model and training parameters
# -------------------------------------------------------------------------------------------------------------------- #

#model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=nb_classes, aux_loss=None)
model_builder = get_model_constructor(model_name)
model = model_builder(pretrained=False, progress=True, num_classes=nb_classes, aux_loss=None)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0., betas=(0.9, 0.999))
criterion = nn.BCEWithLogitsLoss()
writer = SummaryWriter(log_dir=logdir)

have_gpu = torch.cuda.is_available()
if have_gpu and not restrict_gpu:
	model = model.cuda()
	criterion = criterion.cuda()

sum_loss = 0.0

# -------------------------------------------------------------------------------------------------------------------- #
# Train-val loop
# -------------------------------------------------------------------------------------------------------------------- #

L_steps = epochs * L_train
for epoch in range(1, epochs+1):  # loop over the dataset multiple times

	start_time = time.time()
	sum_batch_time = 0
	sum_loss = 0
	sum_accuracy = 0
	g_step = 0

	for step_train, batch_train in enumerate(loader_train, start=1):
		# DEBUG
		if step_train > 20:
			break

		time_batch = time.time()
		g_step += 1

		# get the inputs; data is a list of [inputs, labels]
		imgs, lblmps = batch_train
		if have_gpu and not restrict_gpu:
			imgs = imgs.cuda()
			lblmps = lblmps.cuda()

		optimizer.zero_grad()
		outputs = model(imgs)['out']
		loss = criterion(outputs, lblmps)
		loss.backward()
		optimizer.step()

		sum_loss += loss.item()

		time_batch = time.time() - time_batch
		sum_batch_time += time_batch
		time_passed = time.time() - start_time
		time_remains = (L_steps - g_step) / g_step * time_passed

		time_batch = time.strftime("%H:%M:%S", time.gmtime(time_batch))
		time_passed = time.strftime("%H:%M:%S", time.gmtime(time_passed))
		time_remains = time.strftime("%H:%M:%S", time.gmtime(time_remains))

		print("[ep: {}/{}] [step: {}/{}] [{}%]".format(epoch, epochs, step_train, L_train,
														int(round(g_step/L_steps)*100)), end='\t')
		print("\t\tTime passed {} (batch time {}, remains {})".format(time_passed, time_batch, time_remains), end='\t')
		print("\t\tLoss {:.4f})\tAcc {}".format(loss.item(), "XXX"))

		# if the time has come, save weights and train log
		if isinstance(ckpt_every_steps, int) and step_train % ckpt_every_steps == 0:
			writer.add_scalar('train/time', sum_batch_time / (step_train * batch_size), g_step)
			writer.add_scalar('train/loss', sum_loss / step_train, g_step)
			save_checkpoint(state=model.state_dict(), save_dir=logdir, epoch="{}_{}".format(epoch, step_train),
							is_best=False)

			"""# TODO
			writer.add_figure('predictions vs. actuals',
							illustrate_model(model, inputs, labels),
							global_step=epoch * L_test + step)
			sum_loss = 0.0
			"""

		# testing
		if isinstance(test_every_steps, int) and step_train % test_every_steps == 0:
			model.eval()

			acc_test = 0
			n_samples = batch_size * L_test
			batch_time_test = 0
			loss_test = 0
			start_time_test = 0

			for step_test, batch_test in enumerate(loader_test, start=1):
				time_batch = time.time()

				imgs, lblmps = batch_test
				imgs = imgs.cuda()
				lblmps = lblmps.cuda()

				outputs = model(imgs)['out']
				loss = criterion(outputs, lblmps)
				loss_test += loss.item()

				time_batch = time.time() - time_batch
				sum_batch_time += time_batch
				time_passed_test = time.time() - start_time_test
				time_remains_test = (L_test - step_test) / step_test * time_passed_test

				time_batch = time.strftime("%H:%M:%S", time.gmtime(time_batch))
				time_passed_test = time.strftime("%H:%M:%S", time.gmtime(time_passed_test))
				time_remains_test = time.strftime("%H:%M:%S", time.gmtime(time_remains_test))

				print("VAL [step: {}/{}] [{}%]".format(step_test, L_test, int(round(step_test/L_test)*100)), end='\t')
				print("Time passed {} (remains {})".format(time_passed_test, time_batch, time_remains), end='\t')
				print("Loss {:.4f})\tAcc {}".format(loss.item(), 'XXX'))

			#
			writer.add_scalar('val/time', batch_time_test / n_samples, global_step=g_step)
			writer.add_scalar('val/loss', loss_test / n_samples, global_step=g_step)
			writer.add_scalar('val/acc', acc_test / n_samples, global_step=g_step)

			print('Testing epoch is done!')

print("Done!")

