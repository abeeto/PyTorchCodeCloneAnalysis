from absl import app
from absl import flags

# torch 
import torch
import torch.optim as optim

# allenNLP
from allennlp.common.file_utils import cached_path

# TODO: I am not sure how to use this PennTreeBank yet :/
from allennlp.data.dataset_readers import PennTreeBankConstituencySpanDatasetReader, UniversalDependenciesDatasetReader, Seq2SeqDatasetReader, LanguageModelingReader
from languages_data.wikitext_dataset_reader import WikiTextDatasetReader
from languages_data.max_len_seq2seq_reader import MaxLengthSeq2SeqReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.models import Model

from allennlp.training.trainer import Trainer
from allennlp.training.checkpointer import Checkpointer
from allennlp.common.util import lazy_groups_of

from languages_data import pos_data_reader, embeddings_factory, iterators_factory, datasets_factory, preprocessing_factory
from language_models import models_factory, datareader_cfg_factory
from languages_predictors import predictors_factory
# distributed
import torch.distributed as dist
import torch.multiprocessing as mlproc
from  train_utils.distributed_trainer import DistributeTrainer

import ops_profiler.flop_counter as counter

import time
import utils as U
import numpy as np
import os
import sys
from urllib.parse import urlparse

import copy

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)

FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, 'The model architecture you want to test')
flags.DEFINE_string('run_name', None, 'The name you want to give to this run')
flags.DEFINE_string('task', None, 'The task the model is trained to do')
flags.DEFINE_string('dataset_dir', 'data/', 'Dataset directory')
flags.DEFINE_string('dataset', 'debug', 'Dataset to use')
flags.DEFINE_string('embeddings', 'basic', 'Embeddings to use')
flags.DEFINE_integer('embeddings_dim', 128, 'Embedding dimension to use')
flags.DEFINE_integer('hiddens_dim', 128, 'Embedding dimension to use')
flags.DEFINE_boolean('use_cuda', False, 'whether to use GPU')
flags.DEFINE_integer('log_interval', 10, 'Batch intervals to log')
flags.DEFINE_integer('batch_size', 16, 'Batch intervals to log')
flags.DEFINE_integer('max_epochs', 1, 'max epoch number to run')
flags.DEFINE_integer('max_vocabs', 100000, 'Maximum number of vocabulary')
flags.DEFINE_string('optimizer', 'adam', 'Gradient descent optimizer')
flags.DEFINE_float('drop_out', 0., 'dropout rate, if it is RNN base: for outputs of each RNN layer except the last layer')
flags.DEFINE_boolean('bidirectional', False, 'if it is RNNbase, whether it becomes bidirectional RNN')
flags.DEFINE_integer('max_len', 40, 'maximum length to generate tokens')
flags.DEFINE_integer('num_layers', 1, 'number of layers of recurrent models')
flags.DEFINE_integer('max_sentence_length', 200, 'maxium length per sentence for the encoder')
flags.DEFINE_bool('profile_only', False, 'Profile the model and exit.')
flags.DEFINE_string('ckpt_dir', '/tmp/ckpt', 'the directory to load and save ckpt')



#TODO: MODEL PARALLEL
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
flags.DEFINE_integer("world_size", 1, "Number of distributed process. e.g. count all gpus in machines.")


flags.mark_flag_as_required('run_name')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('task')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('dataset')

data_reader_factory = {
  'debug': pos_data_reader.TaskDataReader,
  'ptb_tree': PennTreeBankConstituencySpanDatasetReader,
  'ud-eng': UniversalDependenciesDatasetReader,
  'nc_zhen': MaxLengthSeq2SeqReader,
  'wikitext': WikiTextDatasetReader
}

test_sentences = {
  'debug' : 'I am your father',
  'nc_zhen' : 'I am your father',
  'ud-eng': ['I', 'am', 'your', 'father'],
}

optimizers_factory = {
  'adam': optim.Adam,
  'sgd': optim.SGD,
  'rmsprop': optim.RMSprop
}

def main(argv):
  del argv

  
  program_flags = FLAGS.flag_values_dict()
  if FLAGS.dist_method != None:
    distribute_main(program_flags)
  else:
    logger, model, reader, out_feature_key, optimizer, iterator, train_dataset, validation_dataset = pre_init(program_flags)
    # if program_flags['profile_only']:
    #   # language
    #   torch.save(model.state_dict(), program_flags['run_name']+"model.pth")
    #   sys.exit(1)

    single_worker(logger, model, reader, out_feature_key, optimizer, iterator, train_dataset, validation_dataset)
    
def pre_init(program_flags, ngpus_per_node=None):
  logger = U.get_logger(__name__+program_flags['run_name'])
  logger.info("run: %s, specified model: %s, dataset: %s", program_flags['run_name'], program_flags['model'], program_flags['dataset'])
  
  cfgs = datareader_cfg_factory.get_datareader_configs(program_flags['dataset'])
  if cfgs is not None:
    if program_flags['max_sentence_length'] > 0:
      cfgs.update({'max_sentence_length': program_flags['max_sentence_length']})
    reader = data_reader_factory[program_flags['dataset']](**cfgs)
  else:
    reader = data_reader_factory[program_flags['dataset']]()

  dataset_paths = datasets_factory.get_dataset_paths(program_flags['dataset'])
  # NOTE: check whether we need preprocessing, i.e. machine translation datasets
  train_dataset = None
  validation_dataset = None
  if dataset_paths['train']['preprocess']:
    # TODO: not yet ready. .___.
    preprocessor = preprocessing_factory.get_preprocessor(program_flags['dataset'])
  
  cache_dataset_dir = os.path.join(program_flags['dataset_dir'], program_flags['dataset'])
  # TODO: If there is multiple paths to make one huge dataset, we should do it with the preprocessor
  train_dataset = reader.read(cached_path(dataset_paths['train']['paths'][0], cache_dataset_dir))
  validation_dataset = None
  if dataset_paths['val'] is not None:
    validation_dataset = reader.read(cached_path(dataset_paths['val']['paths'][0], cache_dataset_dir))
  
  vocab = Vocabulary.from_instances(
                train_dataset + validation_dataset, max_vocab_size=program_flags['max_vocabs'])
  embeddings = embeddings_factory.get_embeddings(program_flags['embeddings'], vocab, embedding_dim=program_flags['embeddings_dim'])
  batch_size = program_flags['batch_size']
  iterator = iterators_factory.get_iterator(program_flags['dataset'], batch_size)
  iterator.index_with(vocab)
  models_args = {
    'model_name': program_flags['model'],
    'embeddings': embeddings,
    'vocab': vocab,
    'input_dims': program_flags['embeddings_dim'],
    'hidden_dims': program_flags['hiddens_dim'],
    'batch_first': True,
    'dataset_name': program_flags['dataset'],
    'dropout': program_flags['drop_out'],
    'bidirectional': program_flags['bidirectional'],
    'max_len': program_flags['max_len'],
    'num_layers': program_flags['num_layers']
  }

  out_feature_key, model = models_factory.get_model_fn(**models_args)

  optimizer = optimizers_factory[program_flags['optimizer']](model.parameters(), lr=0.001)
  return logger, model, reader, out_feature_key, optimizer, iterator, train_dataset, validation_dataset


def distribute_main(program_flags):
  if program_flags['discover_gpus']:
    ngpus_per_node = torch.cuda.device_count()
  else:
    ngpus_per_node = program_flags['num_gpus']

  world_size = program_flags['world_size']
  mlproc.spawn(distribute_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, world_size, program_flags))

def distribute_worker(gpu_index, ngpus_per_node, world_size, program_flags):
  
  # at this point, rank is just machine rank.
  rank = program_flags['rank']

  ckpter = None
  if rank == 0:
    # only ever first rank do the ckpt to save time.
    ckpter = Checkpointer(serialization_dir=program_flags['ckpt_dir'], num_serialized_models_to_keep=2)

  if world_size > 1:
    # NOTE: however here, we need to convert rank to be global rank among processes
    # machine * gpus per node + our current gpu index
    # see https://github.com/pytorch/examples/blob/master/imagenet/main.py
    if program_flags['assume_same_gpus']:
      rank = rank * ngpus_per_node + gpu_index
    else:
      rank = rank * program_flags['rank_scale_factor'] + gpu_index
    
  program_flags['run_name'] = program_flags['run_name'] + str(rank)
  logger, model, reader, out_feature_key, optimizer, iterator, train_dataset, validation_dataset = pre_init(program_flags, ngpus_per_node)

  dist.init_process_group(backend=program_flags['dist_backend'], init_method=program_flags['dist_method'], world_size=world_size, rank=rank)
  
  logger.info("Rank %d --- preparing to start training", rank)
  # Set cuda to a single gpu context  
  torch.cuda.set_device(gpu_index)
  device = torch.device("cuda:%d" % gpu_index)
  model.cuda(gpu_index)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_index])

  trainer = DistributeTrainer(rank=rank, 
                              worldsize=world_size, 
                              ngpus_per_node=ngpus_per_node, 
                              cuda_device=[gpu_index],
                              model=model, 
                              optimizer=optimizer, 
                              iterator=iterator,
                              train_dataset=train_dataset,
                              validation_dataset=validation_dataset,
                              serialization_dir=program_flags['ckpt_dir'],
                              checkpointer=ckpter,
                              log_batch_size_period=20,
                              )
                              
  logger.info(device)
  start_time = time.time()
  trainer.train()
  final_time = time.time() - start_time
  logger.info("Rank %d Finished training: ran for %d secs", rank, final_time)
  if rank == 0:
    # only 1 worker need to do an output check.
    final_output(program_flags, model, device, logger, reader, out_feature_key, start_time)

def single_worker(logger, model, reader, out_feature_key, optimizer, iterator, train_dataset, validation_dataset):
  
  _cudart = U.get_cudart()
  device = torch.device("cuda" if FLAGS.use_cuda else "cpu")
  if _cudart is None:
    logger.warning("No cudart, probably means you do not have cuda on this machine.")
  model = model.to(device)
  cuda_device = 0 if FLAGS.use_cuda else -1 # TODO: multi GPU
  # NOTE: THIS CKPT Mechanism only ckpt at the end of every epoch.
  # if an epoch is more than 1 day, then you take care of it yourself :P
  ckpter = Checkpointer(serialization_dir=FLAGS.ckpt_dir, num_serialized_models_to_keep=1)
  
  if FLAGS.profile_only:
    raw_train_generator = iterator(train_dataset, num_epochs=1, shuffle=False)
    train_generator = lazy_groups_of(raw_train_generator, 1)
    _prof_input = next(train_generator)[0]
    stats = counter.profile(model, input_size=(FLAGS.batch_size,), logger=logger, is_cnn=False, rnn_input=_prof_input)
    logger.info("DNN_Features: %s", str(stats))
    sys.exit(0)

  trainer = Trainer(model=model,
                    optimizer=optimizer,
                    iterator=iterator,
                    train_dataset=train_dataset,
                    serialization_dir=FLAGS.ckpt_dir,
                    validation_dataset=validation_dataset,
                    num_epochs=FLAGS.max_epochs,
                    checkpointer=ckpter,
                    log_batch_size_period = 10,
                    cuda_device=cuda_device)
                    
  start_time = time.time()
  try:
    status = None
    if _cudart:
      status =  _cudart.cudaProfilerStart()
    trainer.train()
  finally:
    if status == 0:
      _cudart.cudaProfilerStop()
  final_time = time.time() - start_time
  logger.info("Finished training: ran for %d secs", final_time)
  final_output(FLAGS.flag_values_dict(), model, device, logger, reader, out_feature_key, start_time)


def final_output(program_flags, model, device, logger, reader, out_feature_key, start_time):
  # TODO: VERY ROUGH.
  if program_flags['task'] == 'lm':

    if "lstm" not in program_flags['model']:
      # TODO: transformer can not use generate() below, not sure why :/
      return

    for _ in range(50):
      bidir_state = 2*program_flags['num_layers'] if program_flags['bidirectional'] else program_flags['num_layers']
      state = (torch.zeros(bidir_state, 1, program_flags['hiddens_dim']).to(device),
        torch.zeros(bidir_state, 1, program_flags['hiddens_dim']).to(device))

      tokens, _ = model.generate(device, state)
      logger.info("GENERATED WORDS:")
      logger.info(' '.join(token.text for token in tokens))
  else:
    model.eval()
    predictor = predictors_factory.get_predictors(program_flags['dataset'], model, reader)
    test_tokens_or_sentence = test_sentences[program_flags['dataset']]
    pred_logits = predictor.predict(test_tokens_or_sentence)
    pred_logits_key = predictors_factory.get_logits_key(program_flags['task'])
    logger.info(pred_logits)
    if pred_logits_key is not None:
      pred_logits = pred_logits[pred_logits_key]

    if program_flags['task'] == 'pos':
      top_ids = np.argmax(pred_logits, axis=-1)
      print([model.vocab.get_token_from_index(i, out_feature_key) for i in top_ids])
    else:
      pred_logits["predictions"] = np.asarray(pred_logits["predictions"])
      logger.info(model.decode(pred_logits))
  
if __name__ == "__main__":
  app.run(main)
