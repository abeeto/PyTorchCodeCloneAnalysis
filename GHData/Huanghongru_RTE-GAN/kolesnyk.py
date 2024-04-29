import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import SNLI, Multi30k
from torchtext.data import Field, BucketIterator

import nltk

import random
import math
import time

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tokenize(text):
    return nltk.tokenize.word_tokenize(text)

TEXT = Field(tokenize = tokenize,
        init_token = '<sos>',
        eos_token = '<eos>',
        include_lengths = True,
        lower=True)

LABEL = Field(tokenize = tokenize,
        lower=True)

train_data, valid_data, test_data = SNLI.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, min_freq=2,
            vectors='glove.42B.300d')
LABEL.build_vocab(train_data, min_freq=2)

BATCH_SIZE = 32

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key = lambda x : x.label,
        device=device)
print "Preparing data completed!"

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)


    def forward(self, src):
        embedded = self.dropout(self.embedding(src))

        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell


class seq2seq(nn.Module):
    def __init__(self, name, encoder, decoder, device):
        super(seq2seq, self).__init__()
        self.name = name

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
                "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0,:]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(TEXT.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 128
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = seq2seq("entailment", enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'weight' in name:
            # nn.init.normal_(param.data, mean=0, std=0.01)
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
        else:
            pass
                            
model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

PAD_IDX = TEXT.vocab.stoi[u'<pad>']

criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train() # ???

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, src_len = batch.premise
        trg, trg_len = batch.hypothesis
        label = batch.label[0,0]

        # print model.name, LABEL.vocab.itos[label]
        if model.name != LABEL.vocab.itos[label]:
            continue

        optimizer.zero_grad()

        output = model(src, trg)

        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        if (i+1) % 500 == 0:
            print "batch %d / %d - loss: %.8f" % (i+1, len(iterator), loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    evaled = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if evaled: break

            src, src_len = batch.premise
            trg, trg_len = batch.hypothesis
            label = batch.label[0,0]
            if model.name != LABEL.vocab.itos[label]: continue
            else: evaled = 1

            output = model(src, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

N_EPOCHS = 1
CLIP = 1

def trainIter():
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model/kolesnyk-%s.pt' % model.name)

        print 'Epoch: %s' % epoch
        print '\tTrain Loss: %.3f' % train_loss
        print '\t Val. Loss: %.3f' % valid_loss

def test():
    model.load_state_dict(torch.load("model/kolesnyk-%s.pt" % model.name))
    model.eval()

    print "load pretrained model completed ! "

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src_len = batch.premise
            trg_len = batch.hypothesis

            rand_col = random.choice(range(src.shape[1]))
            rand_input = src[:,rand_col]
            print "Premise: %s" % visualSent(rand_input)

            output, attns = model(src, trg, 0)

            rand_output = output[1:, rand_col, :].squeeze()
            print "Gen hyp: %s" % visualSent(rand_output.argmax(dim=1))
            print "Ori hyp: %s" % visualSent(trg[:, rand_col])
            print "Label: %s" % LABEL.vocab.itos[batch.label[0,rand_col].item()]

            print "\n"

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(test_iterator)

def visualSent(word_idxs):
    sent = [TEXT.vocab.itos[idx] for idx in word_idxs if idx not in [1,2,3]]
    return " ".join(sent)

# sent, outs = visualSent(u'a lazy fox jumps over a cute dog.')
trainIter()
