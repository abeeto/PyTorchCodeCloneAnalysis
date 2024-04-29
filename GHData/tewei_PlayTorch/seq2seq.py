import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from plots import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
#UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.w2idx = {}
        self.w2cnt = {}
        self.idx2w = {0: "SOS", 1: "EOS"}
        self.vocab_size = 2
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.w2idx:
            self.w2idx[word] = self.vocab_size
            self.w2cnt[word] = 1
            self.idx2w[self.vocab_size] = word
            self.vocab_size += 1
        
        else:
            self.w2cnt[word] += 1

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(l1, l2, filename='deu.txt', rev=False):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if rev:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(l2)
        output_lang = Lang(l1)
    else:
        input_lang = Lang(l1)
        output_lang = Lang(l2)

    return input_lang, output_lang, pairs

MAX_LEN = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LEN and len(p[1].split(' '))  < MAX_LEN and p[1].startswith(eng_prefixes)

def prepareData(l1, l2, filename="deu.txt", rev=False):
    input_lang, output_lang, pairs = readLangs(l1, l2, filename, rev)
    pairs = [p for p in pairs if filterPair(p)]
    print("Trimmed pairs: %d" % (len(pairs)))
    for p in pairs:
        input_lang.addSentence(p[0])
        output_lang.addSentence(p[1])
    
    print("Vocab sizes: %d, %d" %(input_lang.vocab_size, output_lang.vocab_size))
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'deu', 'data/deu.txt', True)
print(random.choice(pairs))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=False)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden) 
        #GRU input / output dim is (batch, seq, feature) when batch_first=True
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#Simple Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#Bahdanau Attention Decoder RNN
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, drop=0.1, max_length=MAX_LEN):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop = drop
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1) 
        #for teacher forcing, the length here is 1 (word by word)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #a_ij = softmax(e_ij) where e_ij = a(s_i-1, h_j)
        #(1, 2*H)->(1, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        #ci = sum(a_ij*h_j)
        #(1, 1, seq_len)*(1, seq_len, H) = (1, 1, H)
        output = torch.cat((embedded[0], context[0]), 1) #(1, 2*H)
        
        #one additional layer
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output) #(1, 1, H)
        
        #feed last hidden state and combined last output/context into GRU
        output, hidden = self.gru(output, hidden)
        #y_i, s_i = GRU(s_i-1, combine(y_i-1, c_i))

        #final output word for time t
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#Luong General Attention
class GenAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, drop=0.1, max_length=MAX_LEN):
        super(GenAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop = drop
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop)
        self.gru = nn.GRU(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)

    def score(self, hidden, encoder_output):
        #use bilinear instead?
        energy = self.attn_w(encoder_output)
        energy = torch.dot(torch.flatten(hidden), torch.flatten(energy))
        #score(h_t, h'_s) = h_t dot (W*h'_s)
        return energy

    def attn(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = torch.zeros(seq_len, device=device)
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        return F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0) #(1, 1, seq_len)

    def forward(self, input, last_context, last_hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1) #word by word (for teacher forcing)
        embedded = self.dropout(embedded)
        rnn_input = torch.cat((embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        #h_t = GRU([x_t, c_t-1]|h_t-1)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0)) #smth wrong
        #a_t(s) = softmax(score(h_t, h'_s)) attention for target->source
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context),1)), dim=1)
        #p(y_t|.., x) = softmax(out([h_t, c_t]))
        #final output word for time t
        return output, context, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    return [lang.w2idx[w] for w in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1,1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_r = 0.5

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_opt, decoder_opt, criterion, max_length=MAX_LEN):
    encoder_hidden = encoder.initHidden() #zero for 1st token
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #generate list of encoder outputs
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_context = torch.zeros(1, decoder.hidden_size, device=device) #not needed for B-attention
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_r else False

    if use_teacher_forcing:
        for di in range(target_length):
            #For B-attention
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #For L-attention
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs) 
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] #teacher forcing

    else:
        for di in range(target_length):
            #For B-attention
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #For L-attention
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs) 
            val, idx = decoder_output.topk(1)
            decoder_input = idx.squeeze().detach() #get rid of reference
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()

    return loss.item()/target_length

def train(encoder, decoder, n_iters, lr=0.0001, prt_c=1000, plt_c=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    #sample or choice?
    criterion = nn.NLLLoss() #negative log likelihood?
    for iter in range(1, n_iters+1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train_step(input_tensor, target_tensor, encoder, decoder, encoder_opt, decoder_opt, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % prt_c == 0:
            print_loss_avg = print_loss_total / prt_c
            print_loss_total = 0
            print('%s (%d epochs %d%% done) loss = %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plt_c == 0:
            plot_loss_avg = plot_loss_total / plt_c
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    showPlot(plot_losses)

def eval(encoder, decoder, sentence, max_length=MAX_LEN):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        #same as training
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoder_context = torch.zeros(1, decoder.hidden_size, device=device) #not needed for B-attention

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs) 
            
            decoder_attentions[di] = decoder_attention.data
            val, idx = decoder_output.data.topk(1) #.data ?
            decoder_input = idx.squeeze().detach() #get rid of reference
            if decoder_input.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.idx2w[idx.item()])

        return decoded_words, decoder_attentions[:di+1]

def evalRand(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)

        print('DE ', pair[0])
        print('EN ', pair[1])
        output_words, attentions = eval(encoder, decoder, pair[0])
        output_sent = ' '.join(output_words)
        print('Output ', output_sent)
        print('-')


hidden_size = 128
encoder = EncoderRNN(input_lang.vocab_size, hidden_size).to(device)
attn_decoder = GenAttnDecoderRNN(hidden_size, output_lang.vocab_size).to(device)
#train(encoder, attn_decoder, 75000, prt_c=5000)
#torch.save(encoder.state_dict(), 'seq2seq_encoder.pt')
#torch.save(attn_decoder.state_dict(), 'seq2seq_attn_decoder.pt')
encoder.load_state_dict(torch.load('seq2seq_encoder.pt'))
attn_decoder.load_state_dict(torch.load('seq2seq_attn_decoder.pt'))
evalRand(encoder, attn_decoder)

output_words, attentions = eval(encoder, attn_decoder, "ich komme mit tom .")
plt.matshow(attentions.numpy())
plt.show()
def evaluateAndShowAttention(input_sentence):
    output_words, attentions = eval(encoder, attn_decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions.numpy())

evaluateAndShowAttention("du bist sehr mutig .")