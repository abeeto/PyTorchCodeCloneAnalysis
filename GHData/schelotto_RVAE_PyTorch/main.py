import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os

from tqdm import tqdm
from model.VAE import RVAE
from dataset import load_data
from torchtext.data import Field
from itertools import chain

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Recurrent VAE')
    parser.add_argument('-embed_dim', type=int, default=300, help='Dimension of word embeddings. [default: 300]')
    parser.add_argument('-rnn_dim', type=int, default=300, help='Dimension of RNN output. [default: 300]')
    parser.add_argument('-num_layer', type=int, default=1, help='Number of RNN layer(s). [default: 1]')
    parser.add_argument('-z_dim', type=int, default=300, help='Dimension of hidden code. [default: 300]')
    parser.add_argument('-p', type=float, default=0.3, help='Dropout probability. [default: 0.3]')
    parser.add_argument('-bidirectional', default=False, action='store_true', help='Use bidirectional RNN.')
    parser.add_argument('-word_emb', type=str, default='none',
                        help='Word embedding name. In none, glove_840B, glove_6B or glove_42B')
    parser.add_argument('-save-file', type=str, default=None, help='File path/name for model to be saved.')
    parser.add_argument('-log-interval', type=int, default=1000, help='Number of iterations to sample generated sentences')
    parser.add_argument('-train_iter', type=int, default=50000, help='Number of iterations for training')
    parser.add_argument('-max_len', type=int, default=60, help='Max length of generated sample sentence')
    args = parser.parse_args()

    (train_iter, valid_iter, _), text_field = load_data(word_emb=args.word_emb, max_len=args.max_len)
    args.vocab_size = len(text_field.vocab.stoi)
    print('Vocabulary size: {}'.format(args.vocab_size))
    args.sos = text_field.vocab.stoi['<sos>']
    args.eos = text_field.vocab.stoi['<eos>']
    args.pad = text_field.vocab.stoi['<pad>']
    args.unk = text_field.vocab.stoi['<unk>']

    rvae = RVAE(args)

    kld_start_inc = 5000
    kld_max = 1
    kld_weight = 0.001
    kld_inc = (kld_max - kld_weight) / (args.train_iter - kld_start_inc)

    if args.word_emb != 'none':

        ###########################
        ## ASSIGN WORD EMBEDDING ##
        ###########################

        rvae.decoder.text_embedder.weight.data = text_field.vocab.vectors.data
        rvae.encoder.text_embedder.weight.data = text_field.vocab.vectors.data

        rvae.decoder.text_embedder.weight.requires_grad = False
        rvae.encoder.text_embedder.weight.requires_grad = False

    ###########################
    ## ASSIGN ADAM OPTIMIZER ##
    ###########################

    update_params = filter(lambda x:x.requires_grad,  chain(rvae.encoder.parameters(), rvae.decoder.parameters()))
    optim = torch.optim.Adam(update_params, lr=1e-3)

    if torch.cuda.is_available():
        rvae = rvae.cuda()

    kld_weight = 0.0001

    for it in range(args.train_iter):
        batch = next(train_iter)
        text = batch.text

        if torch.cuda.is_available():
            text = text.cuda()

        output, kld = rvae(text)

        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        ce_loss = F.cross_entropy(output, text.view(-1), size_average=False)
        kl_loss = kld
        loss = ce_loss + kld_weight * kl_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (it + 1) % args.log_interval == 0:
            rvae.eval()
            print('Iter {}/{} Recon Loss {:.4f} KL Loss {:.4f}'
                  .format(it + 1, args.train_iter, ce_loss.data.item(), kld.data.item()))

            valid_batch = next(valid_iter)
            valid_text = valid_batch.text

            if torch.cuda.is_available():
                valid_text = valid_text.cuda()

            output, kld = rvae(valid_text[0, :].view(1, -1))
            original_indices = valid_text[0, :].data
            generated_indices = torch.max(output, -1)[1].data

            if torch.cuda.is_available():
                original_indices = original_indices.cpu()
                generated_indices = generated_indices.cpu()

            original_indices = original_indices.numpy()
            generated_indices = generated_indices.numpy()

            original_sentence = map(lambda x:text_field.vocab.itos[x], original_indices)
            generated_sentence = map(lambda x:text_field.vocab.itos[x], generated_indices)

            print('Origin: {}'.format(' '.join(filter(lambda x:x!='<pad>', original_sentence))))
            print('Recons: {}'.format(' '.join(filter(lambda x:x!='<pad>', generated_sentence))))

            z = torch.randn(1, args.z_dim)
            if torch.cuda.is_available():
                z = z.cuda()
            sampled_indices = rvae.sample_sentence(z)
            sampled_sentence = map(lambda x:text_field.vocab.itos[x], sampled_indices)
            print('Sample: {}'.format(' '.join(sampled_sentence)))
            rvae.train()

            if not os.path.isdir('trained_model'):
                os.makedirs('trained_model')

            torch.save(rvae.state_dict(), 'trained_model/RVAE.pt')