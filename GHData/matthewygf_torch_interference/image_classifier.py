import os
from absl import app
from absl import flags
# using predefined set of models
import torchvision.datasets as predefined_datasets
from image_models.model import EfficientNet
from image_models.utils import save_ckpt
import image_models.factory as model_factory
import train_utils.data as data_utils

import torch
import torch.optim as optim
import time
import ctypes
import csv
import datetime
import utils as U
import ops_profiler.flop_counter as counter
import ops_profiler.flop_counter_v2 as counter_v2

# distributed
import torch.distributed as dist
import torch.multiprocessing as mlproc

import signal
import sys
from functools import partial

# NOTE: always be deterministic, but means slower in CUDNN.
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FLAGS = flags.FLAGS

flags.DEFINE_string('run_name', None, 'The name you want to give to this run')
flags.DEFINE_string('model', None, 'The model you want to test')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory')
flags.DEFINE_integer('batch_size', 64, 'Batch size of the model training')
flags.DEFINE_string('dataset', 'cifar10', 'The dataset to train with')
flags.DEFINE_boolean('use_cuda', False, 'whether to use GPU')
flags.DEFINE_integer('log_interval', 10, 'Batch intervals to log')
flags.DEFINE_integer('max_epochs', 5, 'maximum number of epochs to run')
flags.DEFINE_bool('profile_only', False, 'profile model FLOPs and Params only, not running the training procedure')
flags.DEFINE_bool('profile_usev2', False, 'profile model FLOPs and Params using another ver., not running the training procedure')
flags.DEFINE_string('ckpt_dir', None, 'directory to save ckpt')
flags.DEFINE_bool('add_random_transforms', False, 'whether to add horizontal flip and vertical flip')
# distributed settings
# NOTE: We use N PROCESS N GPUS distributed method, because its the fastest for pytorch.
flags.DEFINE_integer('num_gpus', 1, "Number of gpus to use within each rank.")
flags.DEFINE_boolean('discover_gpus', False, "if set to true, then will use torch cuda device_count to override num_gpus")
flags.DEFINE_boolean('assume_same_gpus', True, "if set to true, then assume same gpus on other ranks, our local ranks would be (rank * ngpus + gpu index), otherwise, (rank * rank_scale_factor + gpu index)")
flags.DEFINE_integer('rank_scale_factor', 1 , "scale the local rank by this factor, only has effect, if assume_same_gpus is set to false")
flags.DEFINE_integer('rank', 0, "distributed rank , which machine you are. start with 0.")
flags.DEFINE_string("dist_backend", None, "Which distributed backend to use, if defined, then will initialize distribute process group.")
#https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
flags.DEFINE_string("dist_method", None, "Which distributed method to use. e.g. starts with file://path/to/file, env://, tcp://IP:PORT. ")
flags.DEFINE_integer("world_size", 1, "Number of distributed process. e.g. all the GPUs.")
flags.DEFINE_integer('thread_workers', 2, 'Number of threads for data loader')
flags.DEFINE_integer('device', 0, "specify a single GPU device for training.")

flags.mark_flag_as_required('run_name')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('dataset_dir')

datasets_factory = {
  'cifar10': predefined_datasets.CIFAR10,
  'imagenet': None,
  'fashionmnist': predefined_datasets.FashionMNIST,
  'svhn': predefined_datasets.SVHN,
}

datasets_shape = {
  'cifar10': (3, 32, 32),
  'imagenet': (3, 224, 224)
}

datasets_sizes = {
  'cifar10': 10,
  'imagenet': 1000
}

def _compute(device, dataloader, is_train, model, optimizer, loss_op, logger, epoch=None, log_interval=10, non_blocking=False, rank=0):
  epoch_start = time.time()
  for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.to(device, non_blocking=non_blocking), target.to(device, non_blocking=non_blocking)
    if is_train:
      optimizer.zero_grad()
    start_time = time.time()
    pred = model(data)
    if is_train:
      loss = loss_op(pred, target)
      loss.backward()
      optimizer.step()
    time_elapsed = time.time() - start_time
    if epoch is not None:
      if batch_idx == 0:
        logger.info("Rank %d: First step of this epoch: %s", rank, str(datetime.datetime.utcnow()))
      
      if batch_idx == len(dataloader) - 1:
        epoch_elapsed = time.time() - epoch_start
        logger.info("Rank %d: Last step of this epoch: %s, ran for %.4f", rank, str(datetime.datetime.utcnow()), epoch_elapsed)

      if batch_idx % log_interval == 0:
        logger.info("Rank %d: Epoch %d: %d/%d [Loss: %.4f] (%.4f sec/step)", 
                    rank, epoch, batch_idx*len(data), 
                    len(dataloader.dataset), loss.item(), time_elapsed)
    else:
      if batch_idx % log_interval == 0 :
        _, predicted = torch.max(pred.data, 1)
        b_size = target.size(0)
        acc = (predicted == target).sum().item()
        logger.info("Rank %d: [acc: %.4f] (%.4f sec/step)", rank, acc/b_size , time_elapsed)


def compute(logger, model, device, loader, optimizer, loss_op, epoch=None, log_interval=10, is_train=True, rank=0):
  if is_train:
    model.train()
    logger.info("Rank %d: training starts", rank)
    _compute(device, loader, is_train, model, optimizer, loss_op, logger, epoch, log_interval, rank=rank)
  else:
    logger.info("Eval Starts")
    model.eval()
    with torch.no_grad():
      _compute(device, loader, is_train, model, optimizer, loss_op, logger, epoch, log_interval, rank=rank)

def main(argv):
  del argv
  
  # distributed init:
  if FLAGS.dist_backend is not None:
    distributed_main()
  else:
    single_main()


def distributed_main():
  # NOTE: see flags discover gpu
  if FLAGS.discover_gpus:
    ngpus_per_node = torch.cuda.device_count()
  else:
    ngpus_per_node = FLAGS.num_gpus

  # NOTE: we are using ngpus nprocess per node
  # hence we need to adjust the world size to be the following
  world_size = FLAGS.world_size
  proc_flags = FLAGS.flag_values_dict()
  mlproc.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, world_size, proc_flags))

def single_main():
  logger = U.get_logger(__name__+FLAGS.run_name)
  logger.info("run: %s, specified model: %s, dataset: %s", FLAGS.run_name, FLAGS.model, FLAGS.dataset)
  _cudart = U.get_cudart()
  if _cudart is None:
    logger.warning("No cudart, probably means you do not have cuda on this machine.")

  if not FLAGS.profile_only:
    device = torch.device("cuda:"+str(FLAGS.device) if FLAGS.use_cuda else "cpu")
  else:
    device = torch.device("cpu")

  dataset_fn = datasets_factory[FLAGS.dataset]
  dataset_classes = datasets_sizes[FLAGS.dataset]
  try:
    have_ckpt = (FLAGS.ckpt_dir is not None and any("model_state_epoch" in x for x in os.listdir(FLAGS.ckpt_dir)))
  except:
    have_ckpt = False
    
  model = model_factory.get_model(FLAGS.model, FLAGS.dataset, dataset_classes)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  if have_ckpt:
    files = os.listdir(FLAGS.ckpt_dir)
    model_checkpoints = [x for x in files if "model_state_epoch" in x]
    if len(model_checkpoints) > 0: 
      logger.info("**********Found ckpt: %s" % model_checkpoints[-1])
      ckpt_path = os.path.join(FLAGS.ckpt_dir, model_checkpoints[-1])
      try:
        ckpt = torch.load(ckpt_path)
      except EOFError:
        ckpt = None
      if ckpt is not None:
        current_epochs = ckpt['epoch']
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        # Move to device. 
        for state in optimizer.state.values():
          for k, v in state.items():
            if isinstance(v, torch.Tensor):
              state[k] = v.to(device)
        logger.info("******Loaded ckpt")
      else:
        logger.info("Cannot load ckpt")
        current_epochs = 1
    else:
      logger.info("No ckpt found")
      current_epochs = 1
  else:
    logger.info("No ckpt found")
    current_epochs = 1

  model = model.to(device)

  if FLAGS.profile_only:
    stats = counter.profile(model, input_size=(FLAGS.batch_size,) + (datasets_shape[FLAGS.dataset]), logger=logger, is_cnn=True)
    logger.info("DNN_Features: %s", str(stats))
    print("DNN_Features: ", str(stats))
    return

  train_loader, val_lodaer = data_utils.get_standard_dataloader(dataset_fn, FLAGS.dataset_dir, FLAGS.batch_size, download=True)
  
  loss_op = torch.nn.CrossEntropyLoss()
  start_time = time.time()

  try:
    if _cudart is not None:
      status = _cudart.cudaProfilerStart()
    else:
      status = None
    for epoch in range(current_epochs, FLAGS.max_epochs+1):
      compute(logger, model, device, train_loader, optimizer, loss_op, epoch=epoch, is_train=True, rank=0)
      # TODO: currently just ckpt every epoch
      # plus 1 because next time around is inclusive.
      if FLAGS.ckpt_dir is not None:
        save_ckpt(logger, epoch+1, model, optimizer, FLAGS.ckpt_dir)

  finally:
    if status == 0:
      _cudart.cudaProfilerStop()

  final_time = time.time() - start_time
  logger.info("Finished: ran for %d secs", final_time)
    
def worker(gpu_index, ngpus_per_node, world_size, proc_flags):
  # NOTE: this is assumed to be in distributed GPUs
  logger = U.get_logger(__name__+proc_flags['run_name'])
  logger.info("run: %s, specified model: %s, dataset: %s", proc_flags['run_name'], proc_flags['model'], proc_flags['dataset'])
 
  # at this point, rank is just machine rank.
  rank = proc_flags['rank']
  if world_size > 1:
    # NOTE: however here, we need to convert rank to beglobal rank among processes
    # machine * gpus per node + our current gpu index
    # see https://github.com/pytorch/examples/blob/master/imagenet/main.py
    if proc_flags['assume_same_gpus']:
      rank = rank * ngpus_per_node + gpu_index
    else:
      rank = rank * proc_flags['rank_scale_factor'] + gpu_index

  is_chief = rank % ngpus_per_node == 0
  logger.info("Rank %d: using GPU: %d for training, total world size: %d, dist method: %s", rank, gpu_index, world_size, proc_flags['dist_method'])
  
  dist.init_process_group(backend=proc_flags['dist_backend'], init_method=proc_flags['dist_method'], world_size=world_size, rank=rank)

  dataset_fn = datasets_factory[proc_flags['dataset']]
  dataset_classes = datasets_sizes[proc_flags['dataset']]
  model = model_factory.get_model(proc_flags['model'], proc_flags['dataset'], dataset_classes)
  
  # Set cuda to a single gpu context  
  torch.cuda.set_device(gpu_index)
  model.cuda(gpu_index)

  batch_size = proc_flags['batch_size']
  thread_workers = proc_flags['thread_workers']
  logger.info("****number of thread workers to use for data loader %d", thread_workers)
  # per process per distributed data parallel
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_index])
  loss_op = torch.nn.CrossEntropyLoss().cuda(gpu_index)
  optimizer = optim.Adam(model.parameters(), 0.001)

  # load ckpt if ckpt exists.
  ckpt_dir = proc_flags['ckpt_dir']
  have_ckpt = (ckpt_dir is not None and any("model_state_epoch" in x for x in os.listdir(ckpt_dir)))
  location = "cuda:%d" % gpu_index
  device = torch.device(location)
  if have_ckpt:
    files = os.listdir(ckpt_dir)
    model_checkpoints = [x for x in files if "model_state_epoch" in x]
    logger.info("**********Found ckpt: %s" % model_checkpoints[-1])
    # TODO: found the epoch ckpt, for now we just keep one.
    ckpt_path = os.path.join(ckpt_dir, model_checkpoints[-1])
    # NOTE: map model to be loaded to specified single GPU
    ckpt = torch.load(ckpt_path, map_location=location)
    current_epochs = ckpt['epoch']
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optim_state_dict'])
    # Move to device. 
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device)
    logger.info("******Loaded ckpt")
  else:
    logger.info("No ckpt found")
    current_epochs = 1
  logger.info("****Rank %d: starting epoch at %d", rank, current_epochs)

  torch.backends.cudnn.deterministic = True
  dataset_dir = proc_flags['dataset_dir']
  sampler, dist_train_loader, val_loader = data_utils.get_distribute_dataloader(dataset_fn, dataset_dir, batch_size, thread_workers, is_chief)
  logger.info("*****Rank %d: each sampler has %d", rank, len(sampler))
  max_epochs = proc_flags['max_epochs']
  
  for epoch in range(current_epochs, max_epochs):
    sampler.set_epoch(epoch)
    compute(logger, model, device, dist_train_loader, optimizer, loss_op, epoch=epoch, log_interval=proc_flags['log_interval'], is_train=True, rank=rank)

    # NOTE: controversal, only saving ckpt in rank 0, first machine, only first process.
    # assuming the ckpt dir is a nfs mounted for each machine
    if rank == 0:
      save_ckpt(logger, epoch+1, model, optimizer, ckpt_dir)


if __name__ == "__main__":
  app.run(main)

