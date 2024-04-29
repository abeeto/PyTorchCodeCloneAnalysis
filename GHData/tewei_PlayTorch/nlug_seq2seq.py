import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from nltk import word_tokenize
from gensim.models import Word2Vec

from preprocess_seq2seq import *
from plots import *

from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LEN = 100
BATCH_SIZE = 4

def prep_word_vectors(sentences):
    word2vec = Word2Vec.load('emb/w2vwiki2018.model')
    word2idx = {'SOS': 0, 'EOS': 1, 'UNK': 2}
    idx2word = ['SOS', 'EOS', 'UNK']
    word_vectors = []
    vocab_size = 3
    for s in sentences:
        for w in s:
            if (w not in idx2word) and (w in word2vec.wv.vocab):
                idx2word.append(w)
                word2idx[w] = vocab_size
                word_vectors.append(word2vec.wv[w])
                vocab_size += 1
    
    word_vectors = np.concatenate((0.5*np.random.randn(3, word2vec.wv.syn0.shape[1]), np.array(word_vectors)), axis=0)
    #print(word2vec.wv['the'], word_vectors[word2idx['the']], idx2word[word2idx['the']])
    return word2idx, idx2word, word_vectors

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, word_vectors, drop_rate=0.2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(word_vectors.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.out_1 = nn.Linear(4*hidden_size, hidden_size)
        self.drop_1 = nn.Dropout(drop_rate)
        self.out_2 = nn.Linear(hidden_size, output_size)

        #embedding layer
        self.embedding = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(word_vectors))
        #self.embedding.weight.requires_grad = False

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        #GRU input dim is (batch, seq, feature) when batch_first=True
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)

class GenAttn(nn.Module):

    def __init__(self, hidden_size, max_length=MAX_LEN):
        super(GenAttn, self).__init__()
        self.hidden_size = hidden_size*2
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        # Create variable to store attention energies
        attn_energies = torch.zeros(seq_len).to(device)
        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = torch.dot(hidden, energy)
        return energy

#Bahdanau Attention Decoder RNN
class AttnDecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, label_size, output_size, word_vectors, drop=0.6, max_length=MAX_LEN):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop = drop
        self.max_length = max_length
        self.attn = GenAttn(self.hidden_size)
        self.attn_combine = nn.Linear(word_vectors.shape[1]+self.hidden_size*2+label_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.drop1 = nn.Dropout(self.drop)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.drop2 = nn.Dropout(self.drop)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size+label_size+word_vectors.shape[1], self.output_size)
        
        #embedding layer
        self.embedding = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(word_vectors))
        #self.embedding.weight.requires_grad = False

    def forward(self, input, hidden, encoder_outputs, nlg_class):
        embedded = self.embedding(input).view(1, 1, -1)
        #for teacher forcing, the length here is 1 (word by word)
        #embedded = self.dropout(embedded)
        attn_weights = self.attn(hidden.flatten(), encoder_outputs)
        #a_ij = softmax(e_ij) where e_ij = a(s_i-1, h_j)
        #(1, 2*H)->(1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        #ci = sum(a_ij*h_j)
        #(1, 1, seq_len)*(1, seq_len, H) = (1, 1, H)
        output = torch.cat((embedded[0].float(), context[0].float(), nlg_class.unsqueeze(0).float()), 1)
        #one additional layer
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output) #(1, 1, H)
        output = F.relu(self.fc1(output))
        output = self.drop1(output)
        output = F.relu(self.fc2(output))
        output = self.drop2(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(torch.cat((output[0], nlg_class.unsqueeze(0).float(), embedded[0].float()), 1)), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class NLUG_generator():

    def __init__(self):
        self.load_data()
        #load word vectors
        self.word2idx, self.idx2word, word_vectors = prep_word_vectors([s[1]+s[2] for s in self.dataset])
        self.vocab_size = len(self.idx2word)
        self.hidden_size = 256
        self.encoder = EncoderRNN(self.vocab_size, self.hidden_size, self.hidden_size, word_vectors).to(device)
        self.decoder = AttnDecoderRNN(self.vocab_size, self.hidden_size, self.label_size, self.vocab_size, word_vectors).to(device)

    def load_data(self):
        self.label2idx, self.idx2label, _, self.dataset = clean_data()
        self.label_size = len(self.idx2label)

    def sent2tensor(self, sent_tok):
        #word_list = [self.word2idx[w] if w in self.word2idx else UNK_token for w in sent_tok]
        word_list = [self.word2idx[w] for w in sent_tok if w in self.word2idx]
        word_list.append(EOS_token)
        #print(word_list)
        return torch.tensor(word_list, device=device).long().view(-1, 1)

    def label2list(self, labels):
        return torch.tensor([self.label2idx[l] for l in labels], device=device).long()
    
    def label2vec(self, label_list):
        vec = torch.zeros(self.label_size, device=device)
        vec[label_list] = 1.0
        return vec

    def loss_func(self, target, output):
        return -torch.mean(torch.sum(target*torch.log(output) + (1-target)*torch.log(1-output), dim=1))

    def train_test_split(self, val_r=0.05, test_r=0.05, rand=True):
        #total_len = len(self.dataset)
        start = 5000
        total_len = 3000
        rand_seq = np.arange(start, start+total_len)
        if rand == True:
            np.random.shuffle(rand_seq)
        train_size = math.floor((1-val_r-test_r)*total_len)
        val_size = math.floor(val_r*total_len)
        train_seq = rand_seq[0:train_size]
        val_seq = rand_seq[train_size:train_size+val_size]
        test_seq = rand_seq[train_size+val_size:]
        return train_seq, val_seq, test_seq

    def train_step(self, input_tensor, nlg_class, target_tensor, criterion, tf=0.2, max_length=MAX_LEN):
        
        encoder_hidden = self.encoder.initHidden() #zero for 1st token
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        #generate list of encoder outputs
        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size*2, device=device)
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden[-1].unsqueeze(0)
        
        #set teacher forcing
        teacher_forcing_rate = tf
        use_teacher_forcing = True if random.random() < teacher_forcing_rate else False

        if use_teacher_forcing:
            for di in range(target_length):
                #For B-attention
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs, nlg_class)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di] #teacher forcing

        else:
            for di in range(target_length):
                #For B-attention
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs, nlg_class)
                val, idx = decoder_output.topk(1)
                decoder_input = idx.squeeze().detach() #get rid of reference
                loss += criterion(decoder_output, target_tensor[di])
                # if decoder_input.item() == EOS_token:
                #     break

        return loss

    def eval(self, input_tensor, nlg_class, max_length=MAX_LEN):
        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden() #zero for 1st token
            input_length = input_tensor.size(0)
            #generate list of encoder outputs
            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size*2, device=device)
            loss = 0
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden[-1].unsqueeze(0)
            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)
            
            for di in range(max_length):
                #For B-attention
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs, nlg_class)
                val, idx = decoder_output.topk(1)
                decoder_input = idx.squeeze().detach() #get rid of reference
                if decoder_input.item() == EOS_token:
                    break
                else:
                    decoded_words.append(self.idx2word[decoder_input.item()])
            
            return decoded_words, decoder_attentions[:di+1]

    def load_model(self, model_path="nlug_batch_attn.pt"):
        self.encoder.load_state_dict(torch.load('nlug_batch_encode.pt'))
        self.decoder.load_state_dict(torch.load('nlug_batch_decode.pt'))
        
    def train(self, lr=1e-4, contd=True, epochs=100):
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        data_x_seq = [self.sent2tensor(p[1]) for p in self.dataset]
        data_x_class = [self.label2vec(self.label2list(p[0])) for p in self.dataset]
        data_y = [self.sent2tensor(p[2]) for p in self.dataset]
        #print (len(data_x), len(data_y))
        
        train_seq, val_seq, test_seq = self.train_test_split()
        if contd == True:
            self.load_model()
            print('previous model loaded')
            
        #torch_dataset = Data.TensorDataset(data_x, data_y)
        #train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=1, shuffle=True, num_workers=2)
        start = time.time()
        tf = 0.1
        for e in range(epochs):
            epoch_loss = 0
            #train
            tf *= 0.9
            for step in range(math.floor(len(train_seq)/BATCH_SIZE)):
                next_batch = train_seq[step*BATCH_SIZE:(step + 1)*BATCH_SIZE]
                loss = 0
                for s in next_batch:
                    loss += self.train_step(data_x_seq[s], data_x_class[s], data_y[s], criterion, tf)
                
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()
                epoch_loss += loss.item()

            print('%s (%d epochs %d%% done) loss = %.4f' % (timeSince(start, (e+1)/epochs), e+1, e+1/epochs* 100, epoch_loss))
            torch.save(self.encoder.state_dict(), 'nlug_batch_encode.pt')
            torch.save(self.decoder.state_dict(), 'nlug_batch_decode.pt')         
            
            #validate 
            val_bleu_score = 0.0
            for vs in val_seq:
                decoded_words, decoded_attentions = self.eval(data_x_seq[vs], data_x_class[vs])
                truth_words = [self.idx2word[w.item()] for w in data_y[vs]]
                val_bleu_score += sentence_bleu([truth_words], decoded_words)
                print(' '.join(truth_words))
                print(' '.join(decoded_words))

            val_bleu_score /= len(val_seq)
            print('val_bleu_score: %f' % val_bleu_score)
            self.demo('Hello, i was wondering if you can tell me where a good restaurant is to book?', ['request_numberofpeople','request_city', 'request_address'])

    def demo(self, raw_sent, nlg_labels):
        self.load_model()
        sent_tok = word_tokenize(raw_sent)
        input_tensor = self.sent2tensor(sent_tok)
        nlg_class = self.label2vec(self.label2list(nlg_labels))
        decoded_words, decoded_attentions = self.eval(input_tensor, nlg_class)
        print(' '.join(decoded_words))

gen = NLUG_generator()
gen.load_data()
gen.train()
gen.demo('Hello, i was wondering if you can tell me where a good restaurant is to book?', ['request_city', 'request_address'])