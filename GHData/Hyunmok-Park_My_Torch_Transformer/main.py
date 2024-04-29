from config import *
from net import *
from utils import *
from pos_embedding import *

def main(conf):

    word_emb = my_word_emb(len_vocab=5000, hidden_dim=conf.hidden_dim)
    pos_encoding = get_sinusoid_encoding_table(conf.n_seq, conf.hidden_dim)
    pos_encoding = torch.FloatTensor(pos_encoding)
    pos_emb = my_pos_emb(pos_encoding)

    inputs = torch.tensor([
            [3091, 3604,  206, 3958, 3760, 3590,    0,    0],
            [ 212, 3605,   53, 3832, 3596, 3682, 3760, 3590]
    ])

    input_sum = make_sum_inputs(conf, word_emb, inputs, pos_emb)

    dec_inputs = torch.tensor([
        [3091, 3604,  206, 3958, 3760, 3590,    0,    0],
        [ 212, 3605,   53, 3832, 3596, 3682, 3760, 3590]
    ])

    dec_input_sum = make_sum_inputs(conf, word_emb, dec_inputs, pos_emb)

    encoder = ENCODER(conf)
    decoder = DECODER(conf)

    attn_mask = make_attn_mask(inputs, input_sum)
    look_ahead_mask = get_attn_decoder_mask(dec_inputs)

    for i in range(100):
        enc_output = encoder(input_sum, attn_mask)
        dec_output = decoder(dec_input_sum, look_ahead_mask, enc_output, attn_mask)

    print(enc_output, dec_output)
    return 0

if __name__ == "__main__":
    conf, unparsed = get_args()
    main(conf)