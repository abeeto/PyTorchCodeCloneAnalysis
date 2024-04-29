from functools import partial
import inspect
import math
from numpy.testing import assert_allclose
import onnx
import os
import pytest
import torch
import torch.nn as nn

import torchtext
from torchtext.data.utils import get_tokenizer
import time

from onnxruntime import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.experimental import _utils, amp, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options
import _test_commons,_test_helpers


def _load_pytorch_transformer_model(device, dynamic_axes=False, legacy_api=False):
    # Loads external Pytorch TransformerModel into utils
    pytorch_transformer_path = os.path.join('')
    pt_model_path = os.path.join(pytorch_transformer_path, 'pt_model.py')
    pt_model = _utils.import_module_from_file(pt_model_path)
    ort_utils_path = os.path.join(pytorch_transformer_path, 'ort_utils.py')
    ort_utils = _utils.import_module_from_file(ort_utils_path)
    utils_path = os.path.join(pytorch_transformer_path, 'utils.py')
    utils = _utils.import_module_from_file(utils_path)

    # Modeling
    model = pt_model.TransformerModel(28785, 200, 2, 200, 2, 0.2).to(device)
    my_loss = ort_utils.my_loss
    if legacy_api:
        if dynamic_axes:
            model_desc = ort_utils.legacy_transformer_model_description_dynamic_axes()
        else:
            model_desc = ort_utils.legacy_transformer_model_description()
    else:
        if dynamic_axes:
            model_desc = ort_utils.transformer_model_description_dynamic_axes()
        else:
            model_desc = ort_utils.transformer_model_description()


    # Preparing data
    train_data, val_data, test_data = utils.prepare_data(device, 20, 20)
    return model, model_desc, my_loss, utils.get_batch, train_data, val_data, test_data
def batchify(data, bsz, TEXT, device):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def prepare_data(device='cpu', train_batch_size=20, eval_batch_size=20):
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    device = torch.device(device)

    train_data = batchify(train_txt, train_batch_size, TEXT, device)
    val_data = batchify(val_txt, eval_batch_size, TEXT, device)
    test_data = batchify(test_txt, eval_batch_size, TEXT, device)

    return train_data, val_data, test_data

def my_loss(x, target):
    x = x.view(-1, 28785)
    return torch.nn.CrossEntropyLoss()(x, target)

def train_ort_model(epoch = 1):
    device = "cuda"
    ntokens=28785
    bptt = 35
    batch_size = 20
    initial_lr = 0.001
    opts = {'device' : {'id' : device}}
    
    train_data, val_data, test_data = prepare_data(device, 20, 20)
    pt_model_path = os.path.join('pt_model.py')
    pt_model = _utils.import_module_from_file(pt_model_path)
    model = pt_model.TransformerModel(28785, 200, 2, 200, 2, 0.2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 35, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| {} | epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.3f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    device, epoch, batch, len(train_data) // bptt, initial_lr,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

train_ort_model()
