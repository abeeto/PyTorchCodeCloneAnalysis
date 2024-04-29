from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from constants import BATCH_SIZE, best_model, BEST_VAL_LOSS, D_HID, DEVICE, DROPOUT, EMBEDDING_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE, N_HEAD, N_LAYERS, EPOCHS
import torch.nn as nn
import torch.optim as optim
from data_process import data_process, batchify
from transformer_model import TransformerModel
import time
from train import train
from evaluate import evaluate
import math
import copy

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Data Processing
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter, vocab, tokenizer)
val_data = data_process(val_iter, vocab, tokenizer)
test_data = data_process(test_iter, vocab, tokenizer)

train_data = batchify(train_data, BATCH_SIZE)
val_data = batchify(val_data, EVAL_BATCH_SIZE)
test_data = batchify(test_data, EVAL_BATCH_SIZE)

n_tokens = len(vocab)
model = TransformerModel(n_tokens, EMBEDDING_SIZE, N_HEAD, D_HID, N_LAYERS, DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(model, train_data, n_tokens, epoch, criterion, optimizer, scheduler)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < BEST_VAL_LOSS:
        BEST_VAL_LOSS = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()

test_loss = evaluate(best_model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)