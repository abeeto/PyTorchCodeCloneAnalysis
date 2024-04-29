import torch
import random
from argparse import ArgumentParser

from .utils import read_langs, filter_pairs, Progress, loss_plot
from .dataset import SentencePairDataset
from .model import EncoderRNN, AttnDecoderRNN

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

teacher_forcing_ratio = 0.5

SOS_token = 0
EOS_token = 1

filter_dict = dict(
    eng_prefixes = None,
    max_length = 10
)

configs = dict (
    lr = 0.01,
    hidden_size = 256,
    epoch = 150,
    print_every = 200,
    plot_every = 200,
)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-lr, --learning_rate', type=float, default=0.01)
    parser.add_argument('-pf' , '--print-every', type=int, default=200)
    parser.add_argument('-pl', '--plot-every', type=int, default=200)
    parser.add_argument('-h', '--hidden_size', type=int, default=256)

    args = parser.parse_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(lang1, lang2, filename, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, filename, reverse)
    print("Read %s sentence pairs" % len(pairs))

    eng_prefixes = filter_dict["eng_prefixes"]
    max_length = filter_dict["max_length"]
    pairs = filter_pairs(pairs, max_length, eng_prefixes)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(random.choice(pairs))

    return input_lang, output_lang, pairs

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(1)

    loss = 0
    
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([[SOS_token] * batch_size], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # print (f' decoder_input:{decoder_input.shape}')
            # print (f'decoder_hidden:{decoder_hidden.shape}')
            # print (f'encoder_outputs:{encoder_output.shape}')

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output)
            loss += criterion(decoder_output, target_tensor[:, di].squeeze())
            decoder_input = target_tensor[:, di].unsqueeze(0)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # print (f' decoder_input:{decoder_input.shape}')
            # print (f'decoder_hidden:{decoder_hidden.shape}')
            # print (f'encoder_outputs:{encoder_outputs.shape}')

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[:, di].squeeze())


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def train_iters(encoder, decoder, train_loader, configs):

    num_steps = len(train_loader)

    progress = Progress(configs['epoch'], num_steps)

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_every = configs['print_every']
    plot_every = configs['plot_every']
    n_iters = configs['epoch']
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=configs['lr'])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=configs['lr'])

    criterion = nn.NLLLoss()

    for it in range(1, n_iters + 1):
        train_iter = iter(train_loader)
        for idx, (src, target) in enumerate(train_iter):
            input_tensor = src.to(device)
            target_tensor = target.to(device)

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if (idx + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                progress.update(it, idx)
                print (f"{progress} , steps: ({idx + 1} / {num_steps}) epochs: ({it} / {n_iters}), \
                , loss:{print_loss_avg:.3f}")
                
            if it % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        print_loss_total = 0
        
    loss_plot(plot_losses)

def main():

    args = parse_args()

    input_lang, output_lang, pairs = preprocess(lang1="eng", lang2="spa", filename="spa.txt", reverse=False)
    dataset = SentencePairDataset(pairs, 'eng', 'spa')
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    encoder = EncoderRNN(input_lang.n_words, configs['hidden_size']).to(device)
    attn_decoder = AttnDecoderRNN(configs['hidden_size'], output_lang.n_words, dropout_p=0.1).to(device)

    train_iters(encoder, attn_decoder, train_loader, configs)

if __name__ == '__main__':
    main()