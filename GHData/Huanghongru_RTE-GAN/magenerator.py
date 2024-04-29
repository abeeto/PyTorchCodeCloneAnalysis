import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import operator

from Queue import PriorityQueue
from torchtext.datasets import SNLI, MultiNLI
from torchtext.data import Field, BucketIterator

import matplotlib
matplotlib.use('Agg')   # deal with server display issue

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import nltk
from nltk.translate.bleu_score import *

import random


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
# train_data, valid_data, test_data = MultiNLI.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, min_freq=2,
        specials=[u'<esos>', u'<nsos>', u'<csos>'],
        vectors='glove.42B.300d')
LABEL.build_vocab(train_data, min_freq=2)

BATCH_SIZE = 32

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key = lambda x : len(x.premise),
        device=device)
print "Preparing data completed !"

def init_process(src, label):
    for i in range(src.shape[1]):
        if LABEL.vocab.itos[label[0,i]]==u'entailment':
            src[0,i] = TEXT.vocab.stoi[u'<esos>']
        elif LABEL.vocab.itos[label[0,i]] == u'neutral':
            src[0,i] = TEXT.vocab.stoi[u'<nsos>']
        elif LABEL.vocab.itos[label[0,i]] == u'contradiction':
            src[0,i] = TEXT.vocab.stoi[u'<csos>']
    return src

class BeamSearchNode(object):
    def __init__(self, hidden_state, pre_node, word_idx, log_prob, length):
        self.hidden = hidden_state
        self.pre_node = pre_node
        self.word_idx = word_idx
        self.logp = log_prob
        self.len = length

    def eval(self, alpha=1.0):
        # log probability helps numerical stability
        reward = 0
        return self.logp / float(self.len - 1 + 1e-6) + alpha * reward

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):

        embedded = self.dropout(self.embedding(src))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        #outputs = [src sent len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
                        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
                                                
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
                                                                        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        # Times 2 because we use bidrectional encoder
        self.attn = nn.Linear((enc_hid_dim*2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim x 2]
        # mask = [batch size, sent src len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # since the dimension of hidden and encoder_outputs are
        # different, we need to concatenate them
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)
        
        # now energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # now v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask==0, -1e10)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim*2)+emb_dim, dec_hid_dim)

        self.out = nn.Linear((enc_hid_dim*2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs, mask)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        return output, hidden.squeeze(0), a.squeeze(1)
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx, sos_idx, eos_idx):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        if trg is None:
            assert teacher_forcing_ratio == 0, "Must be zero during training"
            inference = True
            trg = torch.zeros((100, src.shape[1])).long().fill_(self.sos_idx).to(self.device)
        else:
            inference = False

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # tensor to store attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_len)

        output = trg[0,:]

        mask = self.create_mask(src)

        for t in range(1, max_len):
            output, hidden, attention = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
            if inference and output.item() == self.eos_idx:
                return outputs[:t], attentions[:t]

        return outputs, attentions

    def beam_search_decode(self, src, src_len, sos_idx, beam_width=3):
        trg = torch.zeros((100, src.shape[1])).long().fill_(sos_idx).to(self.device)
        
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        topk = 1
        trg_vocab_size = self.decoder.output_dim

        encoder_outputs, hidden = self.encoder(src, src_len)

        output = trg[0,:]
        mask = self.create_mask(src)

        # beam search part
        endnodes = []
        number_required = min((topk+1), topk-len(endnodes))

        # starting node - hidden vector, previous node, word id, logp, len
        node = BeamSearchNode(hidden, None, output, 0, 1)
        nodes = PriorityQueue()

        nodes.put((-node.eval(), node))
        qsize = 1

        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, cur = nodes.get()
            decoder_input = cur.word_idx
            decoder_hidden = cur.hidden
            # print "word idx: %s\tscore: %.4f" % (cur.word_idx.item(), score)

            if cur.word_idx.item() == EOS_IDX and cur.pre_node != None:
                endnodes.append((score, cur))
                if len(endnodes) >= number_required: break
                else: continue

            decoder_output, decoder_hidden, attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs, mask)

            log_prob, indexes = torch.topk(F.log_softmax(decoder_output, dim=1), beam_width)
            # print 'log_prob: %s\nindexes: %s' % (log_prob, indexes)

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(-1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, cur, decoded_t, cur.logp+log_p, cur.len+1)

                score = -node.eval()

                # put new nodes into the queue
                nodes.put((score, node))
            qsize += beam_width-1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        decoded_batch = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.word_idx)

            # back tracking
            while n.pre_node != None:
                n = n.pre_node
                utterance.append(n.word_idx)

            utterance = utterance[::-1]
            utterance = [t.item() for t in utterance]
            utterances.append(utterance)

        decoded_batch.append(utterances)

        return decoded_batch

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(TEXT.vocab)
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
ENC_HID_DIM = 256
DEC_HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[u'<pad>']
SOS_IDX = TEXT.vocab.stoi[u'<sos>']
EOS_IDX = TEXT.vocab.stoi[u'<eos>']

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device, PAD_IDX, SOS_IDX, EOS_IDX).to(device)

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
        src = init_process(src, batch.label)
        trg = init_process(trg, batch.label)

        optimizer.zero_grad()

        output, attention = model(src, src_len, trg)

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
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.premise
            trg, trg_len = batch.hypothesis
            src = init_process(src, batch.label)
            trg = init_process(trg, batch.label)

            output, attention = model(src, src_len, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

N_EPOCHS = 16
CLIP = 1

def trainIter():
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        print "validatating ... "
        valid_loss = evaluate(model, valid_iterator, criterion)


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model/masnli-model.pt')

        print 'Epoch: %s' % epoch
        print '\tTrain Loss: %.3f' % train_loss
        print '\t Val. Loss: %.3f' % valid_loss

result_model = 'model/result/signal_trigger_work_20190513.pt'
def test():
    model.load_state_dict(torch.load('model/masnli-model.pt'))
    model.eval()

    print "load pretrained model completed ! "

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src, src_len = batch.premise
            trg, trg_len = batch.hypothesis
            src = init_process(src, batch.label)
            trg = init_process(trg, batch.label)

            rand_col = random.choice(range(src.shape[1]))
            rand_input = src[:,rand_col]
            print "Premise: %s" % visualSent(rand_input)

            output, attns = model(src, src_len, trg, 0)

            rand_output = output[1:, rand_col, :].squeeze()
            print "Gen hyp: %s" % visualSent(rand_output.argmax(dim=1))
            print "Ori hyp: %s" % visualSent(trg[:, rand_col])
            print "Label: %s" % LABEL.vocab.itos[batch.label[0,rand_col].item()]

            beam_test(rand_output)
            print "\n"

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(test_iterator)

def beam_test(output, beam_size=5):
    idxs = output.topk(k=beam_size, dim=1)[1]
    for idx_cand in idxs:
        top1 = idx_cand[0].item()
        if TEXT.vocab.itos[top1] == u'<eos>':
            break
        print "%13s\t| " % TEXT.vocab.itos[top1],
        for idx in idx_cand[1:]:
            print "%10s" % TEXT.vocab.itos[idx.item()], 
        print

def visualSent(word_idxs):
    sent = [TEXT.vocab.itos[idx] for idx in word_idxs if idx not in [1,2,3]]
    return " ".join(sent)

def generate_sentence(model, sentence):
    model.load_state_dict(torch.load('model/masnli-model.pt'))
    model.eval()

    tokenized = tokenize(sentence)
    tokenized = [u'<sos>'] + [t.lower() for t in tokenized] + [u'<eos>']
    numericalized = [TEXT.vocab.stoi[t] for t in tokenized]

    sentence_len = torch.LongTensor([len(numericalized)]).to(device)
    input_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)

    outputs, attentions = model(input_tensor, sentence_len, None, 0)

    outputs = torch.argmax(outputs.squeeze(1), 1)

    output = [TEXT.vocab.itos[t] for t in outputs]

    output = output[1:]
    attentions = attentions[1:]
    return output, attentions


def display_attention(candidate, translation, attention, fig_name):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    attention = attention.squeeze(1).cpu().detach().numpy()
    cax = ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in tokenize(candidate)] + ['<eos>'], 
                        rotation=45)
    ax.set_yticklabels([''] + translation)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    fig.savefig(fig_name)
    plt.close()

def attn_test(data, example_idx, fig_name):
    premise = ' '.join(vars(data.examples[example_idx])['premise'])
    hypothesis = ' '.join(vars(data.examples[example_idx])['hypothesis'])

    gen_hyp, attn = generate_sentence(model, premise)

    display_attention(premise, gen_hyp, attn, fig_name)

def generate_bs_sentence(model, dataset, example_idx, label=None):
    model.load_state_dict(torch.load('model/masnli-model.pt'))
    model.eval()

    sentence = ' '.join(vars(dataset.examples[example_idx])['premise'])
    label = vars(dataset.examples[example_idx])['label'][0] if label is None else label
    signals = {u'entailment': u'<esos>', u'contradiction': u'<csos>', u'neutral': u'<nsos>'}
    if label not in [u'entailment', u'contradiction', u'neutral']:
        print "Bad example..."
        return [[[]]]

    # print sentence

    tokenized = tokenize(sentence)
    tokenized = [signals[label]] + [t.lower() for t in tokenized] + [u'<eos>']
    numericalized = [TEXT.vocab.stoi[t] for t in tokenized]

    sentence_len = torch.LongTensor([len(numericalized)]).to(device)
    input_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)

    decoded_batch = model.beam_search_decode(input_tensor, 
            sentence_len, TEXT.vocab.stoi[signals[label]])

    return decoded_batch

def bleu(dataset):
    gen_hyps = []
    goldens = []
    signals = {u'entailment': u'<esos>', u'contradiction': u'<csos>', 
            u'neutral': u'<nsos>', u'-': ''}
    for i in range(len(dataset.examples)):
        cur_data = vars(dataset.examples[i])
        label = signals[cur_data['label'][0]]
        if not label: continue

        gen_hyp = generate_bs_sentence(model, test_data, i)[0][0]
        gen_hyp = visualSent(gen_hyp).split()

        golden = [label] + vars(dataset.examples[i])['hypothesis']

        gen_hyps.append(gen_hyp)
        goldens.append([golden])

        print "Premise: %s" % (' '.join(cur_data['premise']))
        print "Gen hyp: %s" % (' '.join(gen_hyp))
        print "Golden hyp: %s\n" % (' '.join(golden))

    chencherry = SmoothingFunction()    # the smoothing method by chen et al.
    bleu_score = corpus_bleu(goldens, gen_hyps, smoothing_function=chencherry.method2)
    print "bleu: %.4f" % bleu_score
    return bleu_score


# sent = generate_bs_sentence(model, train_data, 32)[0][0]
# print visualSent(sent)

# o, a = generate_sentence(model, "a little league team tries to catch a runner sliding into a base in an afternoon game .")
# print o

# trainIter()
# print "Test loss: %.4f" % test()

# attn_test(train_data, 56, 'attn1.png')
# attn_test(train_data, 56)

bleu(test_data)

