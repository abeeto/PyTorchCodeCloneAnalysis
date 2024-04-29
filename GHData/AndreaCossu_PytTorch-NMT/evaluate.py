import random
import torch
from torch.autograd import Variable
from utils import variable_from_sentence, show_attention


def validation_loss(input_variable, target_variable, encoder, decoder, criterion, pars):
    loss = 0
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        # Getting size of target sentences
        target_length = target_variable.size()[0]
    
        # Running words through Encoder
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
        # Preparing input and output variables
        decoder_input = Variable(torch.LongTensor([[pars.sos_token]]))
        decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
        if pars.USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        # Use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0].unsqueeze(0), target_variable[di])

            # Getting most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
            if pars.USE_CUDA:
                decoder_input = decoder_input.cuda()

            # Stopping at end of sentence
            if ni == pars.eos_token:
                break
    
        return loss.data.item() / target_length


def evaluate(sentence, encoder, decoder, pars):
    encoder.eval()
    decoder.eval()

    # Converting sentence pairs to indexes variables
    input_variable = variable_from_sentence(pars.input_lang, sentence, pars)

    # Running words through Encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Creating starting vectors for Decoder
    decoder_input = Variable(torch.LongTensor([[pars.sos_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if pars.USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden # Use last hidden state from Encoder to initialize Decoder

    # Preparing output variables
    decoded_words = []
    decoder_attentions = torch.zeros(pars.max_length, pars.max_length)

    # Running through Decoder
    for di in range(pars.max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choosing top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == pars.eos_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(pars.output_lang.index2word[ni.item()])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if pars.USE_CUDA:
            decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def evaluate_randomly(encoder, decoder, pars, pairs):
    # Choosing some random sentence pairs
    pair = random.choice(pairs)

    # Evaluating chosen sentence pairs
    output_words, decoder_attn = evaluate(pair[0], encoder, decoder, pars)
    output_sentence = ' '.join(output_words)

    # Printing result to user
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')

def evaluate_and_show_attention(input_sentence, encoder, decoder, pars):

    # Evaluating given sentence
    output_words, attentions = evaluate(input_sentence, encoder, decoder, pars)

    # Printing result to user
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))

    # Printing attention plot
    show_attention(input_sentence, output_words, attentions)
