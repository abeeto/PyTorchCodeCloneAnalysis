import torch
import torch.nn.functional as F
from positional_encodings import PositionalEncoding1D
START_OF_SEQUENCE_TOKEN = 0


"""class tf_enc_lstm_dec_sender_model(torch.nn.Module):
    def __init__(self, enc_d_model, enc_nhead, enc_dim_ff, enc_dropout=0., enc_num_layers=1, dec_hidden_size=128, dec_layers=2, vocab_size=2000, embedding_depth=128, max_steps=45):
        super().__init__()
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=enc_d_model, nhead=enc_nhead, dim_feedforward=enc_dim_ff, dropout=enc_dropout)
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=enc_num_layers)
        self.max_steps = max_steps

        self.lstm_initial_h_layers = [torch.nn.Linear(2048, dec_hidden_size) for _ in range(dec_layers)]
        self.lstm_initial_c_layers = [torch.nn.Linear(2048, dec_hidden_size) for _ in range(dec_layers)]

        self.decoder = torch.nn.LSTM(input_size=#TODO)

        self.output = torch.nn.Linear(dec_hidden_size, embedding_depth)

        self.embedding = torch.nn.Embedding(vocab_size, embedding_depth)

    def forward(self, x):
        features, target_feature = x
        encoded_features = self.encoder(features)
        initial_h = [F.relu(layer(target_feature)) for layer in self.initial_h_layers]
        initial_h = torch.stack(initial_h)
        initial_c = [F.relu(layer(target_feature)) for layer in self.initial_c_layers]
        initial_c = torch.stack(initial_c)
        state = (initial_h, initial_c)

        token = torch.zeros(size=(encoded_features.shape[0]) + START_OF_SEQUENCE_TOKEN

        for step in range(self.max_steps):
            #compute lstm input based on encoded features, current h and c and last output
            token_embedding = self.embedding(token)
            lstm_input=#TODO
            lstm_out, state = self.decoder(lstm_input, state)

            output_activation = F.relu(self.output(lstm_out))

            logits = torch.matmul(output_activation, torch.transpose(sself.embedding.weight))
            probs = F.softmax(logits)
            sample = torch.distributions.categorical.Categorical(probs=probs).sample(sample_shape=(1,))

            print('use sparse optimizer!!!!!')
"""
def prob_mask(tokens, eos_token=4):
    #only include timesteps before and including endofsequence
    #creates a mask that is 0 for all elements after the eostoken has been reached, 1 else

    #check where you find eos tokens:
    eos_tokens = tokens==eos_token

    #mark everything that is or is after a eos token
    eos_tokens = torch.cumsum(eos_tokens, dim=1)
    #do it twice so the first eos token has a value of 1, all later tokens are > 1
    eos_tokens = torch.cumsum(eos_tokens, dim=1)


    #now we have counts but we want whether count is <=1 as a float (to include the eos token!)

    eos_tokens = (eos_tokens<=1.).to(dtype=float)

    return eos_tokens

class transformer_sender_agent(torch.nn.Module):
    def __init__(self, feature_in_size, vocab_size, embedding_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.encoder_decoder = torch.nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                                    dim_feedforward, dropout, batch_first=True)
        self.img_embedding = torch.nn.Linear(feature_in_size, embedding_size)
        self.seq_embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding1D(embedding_size)


    def forward(self, img_data, seq, device):
        #TODO implement the seq == None case!
        img_embed = self.img_embedding(img_data)
        seq_embed = self.seq_embedding(seq)
        pos_enc = self.pos_encoder(seq_embed)
        seq_embed = seq_embed + pos_enc
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.size()[1])
        mask = mask.to(device=device)
        transformed_seq = self.encoder_decoder(src=img_embed, tgt=seq_embed, tgt_mask=mask)
        logits = torch.matmul(transformed_seq, torch.swapaxes(self.seq_embedding.weight, 0,1))
        return logits

class lstm_sender_agent(torch.nn.Module):
    def __init__(self, feature_size, text_embedding_size, vocab_size, lstm_size, lstm_depth, feature_embedding_hidden_size, start_of_seq_token_idx=3):
        super().__init__()
        self.feature_embedding_hidden = torch.nn.Linear(feature_size, feature_embedding_hidden_size)
        self.feature_embedding_cstate = torch.nn.ModuleList([torch.nn.Linear(feature_embedding_hidden_size, lstm_size) for _ in range(lstm_depth)])
        self.feature_embedding_hstate = torch.nn.ModuleList([torch.nn.Linear(feature_embedding_hidden_size, lstm_size) for _ in range(lstm_depth)])

        self.lstm = torch.nn.LSTM(input_size=text_embedding_size, hidden_size=lstm_size, num_layers=lstm_depth, batch_first=True)
        self.seq_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embedding_size)
        self.seq_embedding_size = text_embedding_size
        self.start_of_seq_token_idx = start_of_seq_token_idx

    def forward(self, img_data, seq_data, device, max_seq_len=40):
        f_embedding = torch.tanh(self.feature_embedding_hidden(img_data))
        c_state = [torch.mean(emb(f_embedding), dim=1) for emb in self.feature_embedding_cstate]
        c_state = torch.stack(c_state, dim=0)
        h_state = [torch.mean(emb(f_embedding), dim=1) for emb in self.feature_embedding_hstate]
        h_state = torch.stack(h_state, dim=0)
        if not seq_data == None:
            #for pretraining
            s_embedding = self.seq_embedding(seq_data)
            lstm_out, _ = self.lstm(s_embedding, (h_state, c_state))
            logits = torch.matmul(lstm_out, torch.swapaxes(self.seq_embedding.weight, 0,1))
            return logits


        else:
            batch_size=img_data.shape[0]
            s_embedding = (torch.zeros(size = (batch_size,1,self.seq_embedding_size)) + self.start_of_seq_token_idx).to(device=device)

            token_agg = []
            log_prob_agg = []

            for t_step in range(max_seq_len):
                lstm_out, (h_state, c_state) = self.lstm(s_embedding, (h_state, c_state))
                logits = torch.matmul(lstm_out, torch.swapaxes(self.seq_embedding.weight, 0,1))
                dist = torch.distributions.categorical.Categorical(logits=logits)
                tokens = dist.sample()
                log_probs = dist.log_prob(tokens)
                s_embedding = self.seq_embedding(tokens)
                token_agg.append(tokens)
                log_prob_agg.append(log_probs)

            tokens = torch.cat(token_agg, dim=1)
            log_probs = torch.cat(log_prob_agg, dim=1)

            return tokens, log_probs







class transformer_receiver_agent(torch.nn.Module):
    def __init__(self, feature_in_size, vocab_size, embedding_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.encoder_decoder = torch.nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                                    dim_feedforward, dropout, batch_first=True)
        self.read_out = torch.nn.Linear(in_features=d_model, out_features=1, bias=False)
        self.img_embedding = torch.nn.Linear(feature_in_size, embedding_size)
        self.seq_embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding1D(embedding_size)

    def forward(self, img_data, seq):
        #TODO implement something to make sure this can't attend timesteps after eos token! (attention mask probably)
        img_embed = self.img_embedding(img_data)
        seq_embed = self.seq_embedding(seq)
        pos_enc = self.pos_encoder(seq_embed)
        seq_embed = seq_embed + pos_enc
        transformed = self.encoder_decoder(src=seq_embed, tgt=img_embed)
        logits = torch.squeeze(self.read_out(transformed))
        return logits

class lstm_receiver_agent(torch.nn.Module):
    def __init__(self, feature_size, text_embedding_size, vocab_size, lstm_size, lstm_depth, feature_embedding_hidden_size, readout_hidden_size, start_of_seq_token_idx=3, end_of_seq_token_idx=4):
        super().__init__()
        self.feature_embedding_hidden = torch.nn.Linear(feature_size, feature_embedding_hidden_size)
        self.feature_embedding_cstate = torch.nn.ModuleList([torch.nn.Linear(feature_embedding_hidden_size, lstm_size) for _ in range(lstm_depth)])
        self.feature_embedding_hstate = torch.nn.ModuleList([torch.nn.Linear(feature_embedding_hidden_size, lstm_size) for _ in range(lstm_depth)])

        self.lstm = torch.nn.LSTM(input_size=text_embedding_size, hidden_size=lstm_size, num_layers=lstm_depth, batch_first=True)
        self.lstm_size = lstm_size
        self.seq_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embedding_size)
        self.seq_embedding_size = text_embedding_size
        self.start_of_seq_token_idx = start_of_seq_token_idx
        self.end_of_seq_token_idx = end_of_seq_token_idx

        self.readout_feature_embedding = torch.nn.Linear(feature_size, lstm_size)
        self.readout_hidden_layer = torch.nn.Linear(2*lstm_size, readout_hidden_size)
        self.readout_layer = torch.nn.Linear(readout_hidden_size, 1)


    def forward(self, img_data, seq):


        f_embedding = torch.tanh(self.feature_embedding_hidden(img_data))
        c_state = [torch.mean(emb(f_embedding), dim=1) for emb in self.feature_embedding_cstate]
        c_state = torch.stack(c_state, dim=0)
        h_state = [torch.mean(emb(f_embedding), dim=1) for emb in self.feature_embedding_hstate]
        h_state = torch.stack(h_state, dim=0)

        s_embedding = self.seq_embedding(seq)
        lstm_out, _ = self.lstm(s_embedding, (h_state, c_state))

        f_out_embedding = torch.tanh(self.readout_feature_embedding(img_data))

        num_features = img_data.size()[1]
        seq_mask = prob_mask(seq)
        #round first to make sure there is no stupid float accuracy issue that shifts the index
        eos_indices = torch.round((torch.sum(seq_mask, dim=1))).long() -1 # -1 because indexing starts at 0 rather than 1... (e.g. the absurd case of eos at tstep zero creates a sum of 1)
        #prepare indices for gather from lstm output by expanding dims via None axes
        eos_indices = eos_indices[:,None, None]
        eos_indices = eos_indices.expand(-1,-1,self.lstm_size)
        eos_indices = eos_indices.clone().detach()
        #print(eos_indices.size())
        eos_lstm_outs = torch.gather(input=lstm_out, dim=1, index=eos_indices)
        #print(eos_lstm_outs.size())
        lstm_out_by_item = eos_lstm_outs.expand(-1,num_features, -1)
        lstm_out_by_item = lstm_out_by_item.clone()
        out_hidden = torch.cat([f_out_embedding, lstm_out_by_item], dim=-1) #concat on feature dim (now shape [batchsize, num_features, lstm_size+hiddenreadoutdepth])
        out_hidden = torch.tanh(self.readout_hidden_layer(out_hidden))
        logits = torch.squeeze(self.readout_layer(out_hidden))
        return logits

    def extract_features(self, img_data, seq):
        f_embedding = torch.tanh(self.feature_embedding_hidden(img_data))
        c_state = [torch.mean(emb(f_embedding), dim=1) for emb in self.feature_embedding_cstate]
        c_state = torch.stack(c_state, dim=0)
        h_state = [torch.mean(emb(f_embedding), dim=1) for emb in self.feature_embedding_hstate]
        h_state = torch.stack(h_state, dim=0)

        s_embedding = self.seq_embedding(seq)
        lstm_out, _ = self.lstm(s_embedding, (h_state, c_state))

        f_out_embedding = torch.tanh(self.readout_feature_embedding(img_data))

        num_features = img_data.size()[1]
        seq_mask = prob_mask(seq)
        #round first to make sure there is no stupid float accuracy issue that shifts the index
        eos_indices = torch.round((torch.sum(seq_mask, dim=1))).long() -1 # -1 because indexing starts at 0 rather than 1... (e.g. the absurd case of eos at tstep zero creates a sum of 1)
        #prepare indices for gather from lstm output by expanding dims via None axes
        eos_indices = eos_indices[:,None, None]
        eos_indices = eos_indices.expand(-1,-1,self.lstm_size)
        eos_indices = eos_indices.clone().detach()
        #print(eos_indices.size())
        eos_lstm_outs = torch.gather(input=lstm_out, dim=1, index=eos_indices)
        #print(eos_lstm_outs.size())
        lstm_out_by_item = eos_lstm_outs.expand(-1,num_features, -1)
        lstm_out_by_item = lstm_out_by_item.clone()
        out_hidden = torch.cat([f_out_embedding, lstm_out_by_item], dim=-1) #concat on feature dim (now shape [batchsize, num_features, lstm_size+hiddenreadoutdepth])
        out_hidden = torch.tanh(self.readout_hidden_layer(out_hidden))
        logits = torch.squeeze(self.readout_layer(out_hidden))
        return out_hidden
