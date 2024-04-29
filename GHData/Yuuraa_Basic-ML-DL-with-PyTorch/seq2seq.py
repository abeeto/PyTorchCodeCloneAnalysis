import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
# English to German datasets
from torchtext.datasets import Multi30k
# Preprocessing
from torchtext.data import Field, BucketIterator
# Tokenizer
import spacy
# Nice loss plots
from torch.utils.tensorboard import SummaryWriter
# model을 로딩하고 이용하는 utils 부분
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load('de')
spacy_neg = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True, init_token = '<sos>', eos_token='<eos>')
german = Field(tokenize=tokenizer_eng, lower=True, init_token = '<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english)) # extensionsand fields

german.build_vocab(train_data, max_size=10000, min_freq = 2) # Once만 나온다면, 이를 vocabulary에 추가 안함 최소 2번은 나와야 추가
english.build_vocab(train_data, max_size=10000, min_freq = 2) # Once만 나온다면, 이를 vocabulary에 추가 안함 최소 2번은 나와야 추가

#TODO: bucket추가해야 함

class Encoder(nn.Module):
    # input_size 이경우 German vocab의 사이즈일 것이고, embedding_size d dimensional
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        # super 클래스란, 자식 클래스에서 부모 클래스의 내용을 사용하고 싶을 경우 사용
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        # Embedding을 하는 부분
        self.embedding = nn.Embedding(input_size, embedding_size)
        # Embedding 된 벡터에 대해 LSTM을 수행할 것이다
        self.rnn = nn.LSTM(embeddign_size, hidden_size, num_layers, dropout=p)
    
    # sentence -> Tokenize된 뒤 -> vocab으로 map -> embedding -> LSTM
    def forward(self, k):
        # x shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)
        
        self.rnn(embedding)
        # hidden과 cell 이 context
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    # output_size가 input_size와 같을 것이다. vector - 단어장 안의 각 단어와 상응하는 벡터를 내보낼 것이다. 각 노드는 그 단어의 vocabulary일 확률을 나타낸다?
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        # embedding size만큼의 인풋을 받은 뒤 hidden_size만한 output으로 매칭한다
        # hidden_size는 encoder와 decoder에서 같은 값이다
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x, hidden, cell):
            # shape of x: (N) but we want (1,N) 왜냐하면 decoder는 각 시간마다 한 개의 word를 예측할 것이기 때문이고, 
            # 이전에 나왔던 단어와 이전 cell 값을 다 받아야 하고, 한 개의 word를 내보내야 함
            x = x.unsqueeze(0) # Add one dimension
            embedding = self.dropout(self.embedding(x))
            # embedding shape: (1, N, embedding_size)

            # send it to the LSTM
            outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
            # shape of outputs: (1, N, hidden_size)

            predictions = self.fc(outputs)
            # shape of predictions: (1, N, length_of_vocab) 이제는 이 1을 없애고 싶다
            predictions = predictions.squeeze(0) # Loss function에서 쓰기 위해서 1을 없앰

            return predictions, hidden, cell


# Combine the Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    # TODO: teacher force ratio 설명 적기. (심플: 실제 답과 예측한 답의 비율. 1이 되면 train에서 본 것과 매우 다른 결과 나올 수 있음)
    def forward(self, source, target, teacher_force_ratio = 0.5):
        batch_size = source.shape[1]
        # (target_length, N)
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        # batch_size, target_vocab_size 가 output의 사이즈
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            # hidden and cell from the encoder (initially?)
            output, hidden, cell = self.decoder(x, hidden, cell)
            ouptuts[t] = output
            # (N, english_vocab_size)
            best_guess = output.argmax(1)
            # Next input to the decoder
            x = target[t] if random.random() < teacher_force_ratio else best_guess
    
        return outputs
### Now we're ready to do the training

# Training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
# Loss가 감소하는 모습을 보여줌
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size = batch_size, sort_within_batch = True, sort_key = lambda x: len(x.src) # 문장들의 길이가 다른데, 비슷한 길이를 더 선호해 -> 패딩 횟수를 줄여서 계산량 줄임
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Encoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
# Padding 한 것에 대해서는 cost function에서 아무 것도 하지 않는다
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        ouput = model(inp_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        # 만약 MNIST dataset이라면 (N,10) 일 것이다.and targets would be (N)
        # trg_len과 batch_size를 
        # TODO
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        # Make sure gradients are in a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step = step)
        step += 1