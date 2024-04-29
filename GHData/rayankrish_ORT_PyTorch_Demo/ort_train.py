import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
import numpy as np

from pt_model import TransformerModel

from onnxruntime.capi.ort_trainer import IODescription, ModelDescription, ORTTrainer
from onnxruntime.capi._pybind_state import set_seed

# set the seeds
torch.manual_seed(0)
set_seed(0)

# Load and batch data
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 20 # thiagofc: original was 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# Functions to generate input and target sequence
bptt = 35
def get_batch(source, i):
    # import pdb; pdb.set_trace()
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# Initiate an instance
ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# Run the model
#lr = 0.001 # learning rate
lr = 0.001

# thiagofc: ORT specific
def my_loss(x, target):
    x = x.view(-1, 28785) #thiagofc: hard-coded for testing
    return nn.CrossEntropyLoss()(x, target)

def transformer_model_description():
    input_desc = IODescription('input1', [bptt, batch_size], torch.float32)
    label_desc = IODescription('label', [bptt, batch_size, ntokens], torch.int64)
    loss_desc = IODescription('loss', [], torch.float32)
    #return ModelDescription([input_desc, label_desc], [loss_desc]), IODescription('Learning_Rate', [lr,], torch.float32)
    prediction_desc = IODescription('prediction', [bptt, batch_size, ntokens], torch.float32)
    return ModelDescription([input_desc, label_desc], [loss_desc, prediction_desc]), IODescription('Learning_Rate', [lr,], torch.float32)

model_desc, lr_desc = transformer_model_description()

def get_lr_this_step(global_step):
    return 1

trainer = ORTTrainer(model, my_loss, model_desc, "LambOptimizer", None, lr_desc, device, _use_deterministic_compute=True)#, get_lr_this_step=get_lr_this_step)
second_trainer = ORTTrainer(model, my_loss, model_desc, "LambOptimizer", None, lr_desc, device, _use_deterministic_compute=True)#, get_lr_this_step=get_lr_this_step)

import time
def train(lr, trainer, data_source, device, epoch):
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt)):
        data, targets = get_batch(data_source, i)
        # print(data.shape)
        # print(targets.shape)
        # import pdb; pdb.set_trace()
        if len(data) < bptt:
            print(f"len(data)={len(data)} < {bptt}")
            continue
        learning_rate = torch.tensor([lr])
        #loss, output = trainer.train_step(data, targets)
        loss, output = trainer.train_step(data, targets, learning_rate)
        loss, output = second_trainer.train_step(data, targets, learning_rate)

        for (a_name, a_vals), (b_name, b_vals) in zip(trainer.session.get_state().items(), second_trainer.session.get_state().items()):
            np_a_vals = np.array(a_vals)
            np_b_vals = np.array(b_vals)
            #print(np.testing.assert_allclose(np_a_vals, np_b_vals, rtol=1e-4))
            print(a_name, np.abs(np_a_vals-np_b_vals).max())


        # save weights to pickle file
        import pickle
        file_name = 'model_run_2.pk'
        outfile = open(file_name, 'wb')
        pickle.dump((trainer.session.get_state(), second_trainer.session.get_state()), outfile)
        outfile.close()
        break
        # import pdb; pdb.set_trace()
        total_loss += loss.item()
        log_interval = 20
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| {} | epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.3f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} |'.format(
                    device, epoch, batch, len(data_source) // bptt, lr,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            print('| torch loss {:5.2f} |'.format(trainer.get_torch_cur_loss()))
            total_loss = 0
            start_time = time.time()

def evaluate(trainer, data_source):
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if len(data) < bptt:
                print(f"len(data)={len(data)} < {bptt}")
                continue
            # import pdb; pdb.set_trace()
            loss = trainer.eval_step(data, targets, fetches=["loss"])
            total_loss += len(data) * loss.item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 1 # The number of epochs

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(lr, trainer, train_data, device, epoch)
    val_loss = evaluate(trainer, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss


# Evaluate the model with the test dataset
test_loss = evaluate(trainer, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
