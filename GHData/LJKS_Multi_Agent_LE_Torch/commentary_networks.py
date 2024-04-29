import torch
class grid_commentary_network(torch.nn.Module):
    def __init__(self, num_senders, num_receivers):
        super().__init__()
        self.num_senders = num_senders
        self.num_receivers = num_receivers
        #I'm completely misusing Embeddings here, sorry PyTorch gods!
        self.commentary_weights = torch.nn.Embedding(num_embeddings=num_senders*num_receivers, embedding_dim=1)
        #Softmax over dim 0 because Embeddings have shape [num_senders*num_receivers,1]
        self.normalize = torch.nn.Softmax(dim=0)

    def forward(self, sender_idx, receiver_idx):
        rel_weights = self.normalize(self.commentary_weights.weight)
        flat_idx = sender_idx*self.num_receivers + receiver_idx
        return torch.squeeze(rel_weights)[flat_idx]







class idx_commentary_network(torch.nn.Module):
    def __init__(self, num_senders, num_receivers, embedding_size, hidden_size):
        super().__init__()
        self.sender_embedding = torch.nn.Embedding(num_embeddings=num_senders, embedding_dim=embedding_size)
        self.receiver_embedding = torch.nn.Embedding(num_embeddings=num_receivers, embedding_dim=embedding_size)
        self.hidden = torch.nn.Linear(2*embedding_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, 1)

    def forward(self, sender_idx_batch, receiver_idx_batch):
        sender_emb = self.sender_embedding(sender_idx_batch)
        receiver_emb = self.receiver_embedding(receiver_idx_batch)
        total_emb = torch.cat([sender_emb, receiver_emb], dim=-1)
        hid = torch.tanh(self.hidden(total_emb))
        commentaries = torch.sigmoid(self.out(hid))
        commentaries = torch.squeeze(commentaries)
        return commentaries

class objects_commentary_network_normalized(torch.nn.Module):
    def __init__(self, num_senders, num_receivers, embedding_size, feature_size, num_encoder_layers, num_decoder_layers, dim_feedforward, nhead, dropout=0.):
        super().__init__()
        self.sender_embedding = torch.nn.Embedding(num_embeddings=num_senders, embedding_dim=embedding_size)
        self.receiver_embedding = torch.nn.Embedding(num_embeddings=num_receivers, embedding_dim=embedding_size)
        self.feature_embedding = torch.nn.Linear(feature_size, 2*embedding_size)
        self.encoder_decoder = torch.nn.Transformer(2*embedding_size, nhead, num_encoder_layers, num_decoder_layers,
                                                    dim_feedforward, dropout, batch_first=True)
        self.readout = torch.nn.Linear(2*embedding_size, 1)


    def forward(self, encoded_tokens, sender_idx_batch, receiver_idx_batch):
        sender_emb = self.sender_embedding(sender_idx_batch)
        receiver_emb = self.receiver_embedding(receiver_idx_batch)
        total_emb = torch.cat([sender_emb, receiver_emb], dim=-1)
        feature_emb = self.feature_embedding(encoded_tokens)

        enc_dec_out = self.encoder_decoder(src=feature_emb, tgt=total_emb)
        pred = torch.squeeze(self.readout(enc_dec_out))
        pred = torch.nn.functional.softmax(pred, dim=-1)
        commentaries = pred
        return commentaries