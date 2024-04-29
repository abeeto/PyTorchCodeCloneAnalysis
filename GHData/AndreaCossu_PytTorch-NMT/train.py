import random
import time
import math
from utils import variables_from_pair
from torch.autograd import Variable
import torch
from utils import show_plot, save_model
from evaluate import validation_loss

teacher_forcing_ratio = 0.5
clip = 5.0


def as_minutes(s):
    # Converting seconds to minutes+seconds
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    # Returning missing time as a string
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, pars):
    # Train model on a sentence pair

    encoder.train()
    decoder.train()

    # Zeroing gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Getting size of target sentences
    target_length = target_variable.size()[0]

    # Running words through Encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Preparing input and output variables
    decoder_input = Variable(torch.LongTensor([[pars.sos_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden  # Use last hidden state from Encoder to initialize Decoder
    if pars.USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choosing whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0].unsqueeze(0), target_variable[di])
            decoder_input = target_variable[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0].unsqueeze(0), target_variable[di])

            # Getting most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if pars.USE_CUDA:
                decoder_input = decoder_input.cuda()

            # Stopping at end of sentence (not necessary when using known targets)
            if ni == pars.eos_token:
                break

    # Backpropagation + gradient clipping to prevent explosion
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item() / target_length


def train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, pars, pairs, val_pairs):
    # Train model on a set of sentence pairs

    start = time.time()
    plot_losses = []
    plot_val_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0   # Reset every plot_every
    val_loss_total = 0    # Initializing validation loss

    print('Starting training...')
    print_every = pars.print_every
    for epoch in range(1, pars.n_epochs + 1):
        # Getting training data for this cycle
        training_pair = variables_from_pair(random.choice(pairs), pars)
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # Running the train function
        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, pars)

        # Keeping track of loss
        print_loss_total += loss
        plot_loss_total += loss

        # Printing missing time to user
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / float(print_every)
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / float(pars.n_epochs)), epoch, epoch / float(pars.n_epochs) * 100, print_loss_avg)
            print(print_summary)
            # eventually save intermediate models
            #save_model(encoder, 'encoder_part')
            #save_model(decoder, 'decoder_part')

        # Adding validation information to plot
        if epoch % pars.plot_every == 0:
            val_iter = 10
            for _ in range(val_iter):
                sentence = variables_from_pair(random.choice(val_pairs), pars)
                val_loss_total += validation_loss(sentence[0], sentence[1], encoder, decoder, criterion, pars)
            val_loss_total /= val_iter
            plot_val_losses.append(val_loss_total)
            val_loss_total = 0
            plot_loss_avg = plot_loss_total / float(pars.plot_every)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # Saving model
    save_model(encoder, 'encoder')
    save_model(decoder, 'decoder')
    print('... training ended.')

    # Printing validation plot
    show_plot(plot_losses, plot_val_losses)
