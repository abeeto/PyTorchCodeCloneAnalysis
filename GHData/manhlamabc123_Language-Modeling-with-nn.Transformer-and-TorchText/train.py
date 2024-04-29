import time
import torch.nn as nn
import math
from constants import BPTT, DEVICE, LEARNING_RATE
from data_process import get_batch

from transformer_model import generate_square_subsequent_mask

def train(model: nn.Module, train_data, n_tokens, epoch, criterion, optimizer, scheduler, learning_rate = LEARNING_RATE) -> None:
    model.train() # Turn on train mode?
    total_loss = 0
    log_interval = 200 # ???
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(BPTT).to(DEVICE)

    num_batches = len(train_data) // BPTT
    for batch, i in enumerate(range(0, train_data.size(0) - 1, BPTT)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != BPTT:
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, n_tokens), targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            learning_rate = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {learning_rate:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')

            total_loss = 0
            start_time = time.time()