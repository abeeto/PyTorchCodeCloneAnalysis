from xmlrpc.client import FastParser
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # we are going to split the embed_size into heads parts. This is why it is MultiAttention.
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert (self.heads_dim * heads ==
                embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        self.fc_out = nn.Linear(
            self.heads * self.heads_dim, self.embed_size, bias=False)

    def forward(self, values, keys, query, mask):
        # This function is general. It will be called in the encoder and decoder.
        # I think the shape of thee values is: (N, D, E): N=Batch dimention, D=number of words for these sentence, E= Embeding number for each word
        # number of examples we send at the same time (batch dimention).
        N = query.shape[0]
        # length of the sequence. They will always be the same value.
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        query = query.reshape(N, query_len, self.heads, self.heads_dim)

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        # 1. MatMult(Q, transpose(K))
        # this looks nice. It seems easier than the other way.
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        # query shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape : (N, heads, query_len, key_len) # with this dimention we can already see the attention weights.

        if mask is not None:
            # where the mask is zero, the value in energy will be -inf. Which means no attention will be given.
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        # the reshape is the concatenation.
        out = torch.einsum('nhql, nlhd->nqhd', [attention, values]).reshape(
            N, query_len, self.heads * self.heads_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after eisum: (N, query_len, heads, head_dim) then flatten last two dimensions

        # out has exactly the same dimensions of the input. (N, sequence_length, Embedings)
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion*embed_size),
                                          nn.ReLU(),
                                          nn.Linear(forward_expansion*embed_size, embed_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        # max_length is the maximum number of words of the biggest sentence.
        self.embed_size = embed_size
        self.device = device
        # word_embeding will be (number_of_words, embeding)
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # 200, 8. For each possible sequence length we have vector.
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x is the input. N is the number of sequences, and seq_length is the number of words.
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        # now we have the position context.
        out = self.dropout(self.word_embedding(
            x) + self.position_embedding(positions))

        for layer in self.layers:
            # because the Q,K and V are the same.
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # src_mask is option
        # trg_mask is the mask necessary due to the autogressive.
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value, key, query, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        # tgt_vocab size is the number of possible words for the target.
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)


        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion,
                          dropout, device) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(
            x) + self.position_embedding(positions)))


        print(f'pos_embedding= {x.shape}')
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        
        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4,
                 heads=8, dropout=0, device='cuda:0', max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))
                              ).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(  # 2 examples
        device
    )
    
    print(x.shape)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [
                       1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    print(trg.shape)    
    
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])  # this is how we would use the transformer. We give the input sequence as x. And then give some part of the target (the translation), and expect the transformer to predict the next word.
    # In this case, since we are giving the trg[:-1], the next word would be the end-of-sentence.
    # Then, the value would be sent to crossentropy loss and so on and so forth.
    print(out.shape)
    # the output is 2x7x10.
    # 2 is the number of sentences
    # 7 is the number of words given for each sentence
    # 10 is because for each word we are predicting the next word. And since trg_vocab_size=10, we need 10 labels for the croosentropy loss.
    
    
    
    
    
    
