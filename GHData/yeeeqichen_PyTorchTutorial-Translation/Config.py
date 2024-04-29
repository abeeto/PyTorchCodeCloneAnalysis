import torch


class Config:
    def __init__(self):
        self.enc_vocab_size = None
        self.enc_embed_size = 128
        self.dec_vocab_size = None
        self.dec_embed_size = 128
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.enc_hidden_size = 256
        self.dec_hidden_size = 256
        self.enc_dropout = 0.2
        self.dec_dropout = 0.2
        self.batch_size = 20
        self.max_len = 10
        self.attn_dim = 8
        self.data_path = 'data/eng-fra.txt'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.lr = 0.001
        pass


config = Config()
