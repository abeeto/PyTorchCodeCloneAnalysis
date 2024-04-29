# Created by  IT-JIM  2021
# Here I fool around with LeCun's transformer
# And try to solve the issues with torchtext version incompatibilities

import sys
import tqdm

import numpy as np
import torch
import torch.utils.data
import torchtext

DEVICE = torch.device('cuda')


########################################################################################################################
def print_it(a, name: str = ''):
    if isinstance(a, torch.Tensor):
        m = a.to(dtype=torch.float32).mean()
    else:
        m = a.mean()

    print(name, a.shape, a.dtype, a.min(), m, a.max())


########################################################################################################################
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model  # All same
        else:
            d_kq, d_xk, d_xv = d_input  # Not used in this code !!!

        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads
        # These are still of dimension d_model. They will be split into number of heads
        self.w_q = torch.nn.Linear(d_xq, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_xk, d_model, bias=False)
        self.w_v = torch.nn.Linear(d_xv, d_model, bias=False)

        # Outputs of all sub-layers need to be of dimension d_model
        self.w_h = torch.nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v):
        # Inputs: (bs, n_heads, q_length, dim_per_head) = [164, 2, 200, 16]
        # batch_size = q.size(0)
        # k_length = k.size(-2)
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        # self.d_k == 16
        q = q / np.sqrt(self.d_k)  # (bs, n_heads, q_length, dim_per_head) = [164, 2, 200, 16]
        scores = torch.matmul(q, k.transpose(2, 3))
        # (bs, n_heads, q_length, k_length) = [164, 2, 200, 200], doubled 200 !
        a = torch.nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length) = [164, 2, 200, 200]

        # Get the weighted average of the values
        h = torch.matmul(a, v)  # [164, 2, 200, 16]
        return h, a

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Input: (batch_size X seq_length X d_model) = [164, 200, 32]
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k) = [164, 2, 200, 16]
        """
        # t = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, x_q, x_k, x_v):
        # All inputs [164, 200, 32]
        batch_size, seq_length, dim = x_q.size()

        # After transforming, split into num_heads
        # Note: w_q does not change dim
        q = self.split_heads(self.w_q(x_q), batch_size)  # (bs, n_heads, q_length, dim_per_head) = [164, 2, 200, 16]
        k = self.split_heads(self.w_k(x_k), batch_size)  # (bs, n_heads, k_length, dim_per_head) = [164, 2, 200, 16]
        v = self.split_heads(self.w_v(x_v), batch_size)  # (bs, n_heads, v_length, dim_per_head) = [164, 2, 200, 16]

        # Calculate the attention weights for each of the heads
        h_cat, a = self.scaled_dot_product_attention(q, k, v)
        # Put all the heads back together by concat
        h_cat = self.group_heads(h_cat, batch_size)
        h = self.w_h(h_cat)
        return h, a


########################################################################################################################
class CNN(torch.nn.Module):
    """A simple MLP d_model -> d_model"""

    def __init__(self, d_model, hidden_dim, p):
        super(CNN, self).__init__()
        self.l1 = torch.nn.Linear(d_model, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, d_model)
        self.a = torch.nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.a(x)
        x = self.l2(x)
        return x


########################################################################################################################
class Embeddings(torch.nn.Module):
    """Returns learnable word emb + positional emb, of size d_model"""

    def __init__(self, d_model, vocab_size, max_position_embeddings, p):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, d_model)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1.e-12)
        self.create_posemb(d_model, max_position_embeddings)

    def create_posemb(self, d_model, max_position_embeddings):
        theta = np.array([
            [p / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
            for p in range(max_position_embeddings)
        ])  # (max_position_embeddings, d_model)
        pp = torch.zeros(max_position_embeddings, d_model, dtype=torch.float32)
        pp[:, 0::2] = torch.tensor(np.sin(theta[:, 0::2]), dtype=torch.float32)
        pp[:, 1::2] = torch.tensor(np.cos(theta[:, 1::2]), dtype=torch.float32)
        for name, p in self.named_parameters():  # position_embeddings.weight
            if name == 'position_embeddings.weight':
                p.requires_grad = False
                with torch.no_grad():
                    p.copy_(pp)
                    # print_it(p, 'p')

    def forward(self, input_ids):
        seq_length = input_ids.size(1)  # 200
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (164, 200)
        # Get word embeddings for each input id
        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim) = (164, 200, 32)
        # Get position embeddings for each position id
        position_embeddings = self.position_embeddings(position_ids)  # (164, 200, 32)
        # Add them both, then Layer norm
        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim) = (164, 200, 32)
        embeddings = self.layer_norm(embeddings)
        return embeddings


########################################################################################################################
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, p)
        self.cnn = CNN(d_model, conv_hidden_dim, p)
        self.layernorm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        # Multi-head attention
        # (batch_size, input_seq_len, d_model) = [164, 200, 32]
        # Size is not changed in here !
        attn_out, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model) = [164, 200, 32]
        # Layer norm after adding the residual connection
        out1 = self.layernorm1(x + attn_out)  # (batch_size, input_seq_len, d_model) = [164, 200, 32]
        # Feed forward
        cnn_out = self.cnn(out1)  # (batch_size, input_seq_len, d_model) = [164, 200, 32]
        # Second layer norm after adding residual connection
        out2 = self.layernorm2(out1 + cnn_out)  # (batch_size, input_seq_len, d_model) = [164, 200, 32]
        return out2


########################################################################################################################
class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, input_vocab_size, max_position_embeddings, p=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embeddings(d_model, input_vocab_size, max_position_embeddings, p)

        self.enc_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, ff_hidden_dim, p))

    def forward(self, x):  # Input:  (batch_size, input_seq_length) = [164, 200]
        x = self.embedding(x)  # Transform to (batch_size, input_seq_length, d_model) = [164, 200, 32]
        for i in range(self.num_layers):  # These layers do not change dimensions !
            x = self.enc_layers[i](x)
        return x  # # (batch_size, input_seq_len, d_model) = [164, 200, 32]


########################################################################################################################
class TransformerClassifier(torch.nn.Module):
    def __init__(self, num_layers=1, d_model=32, num_heads=2, conv_hidden_dim=128, input_vocab_size=50002,
                 num_answers=2):
        super(TransformerClassifier, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, input_vocab_size,
                               max_position_embeddings=10000)
        self.dense = torch.nn.Linear(d_model, num_answers)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        x = self.dense(x)
        return x


########################################################################################################################
def main_stupid1():
    """Test kvq selection"""
    print('haha')
    mha = MultiHeadAttention(512, 8, 0.)
    k = torch.tensor(
        [[10, 0, 0],
         [0, 10, 0],
         [0, 0, 10],
         [0, 0, 10]],
        dtype=torch.float32).view(1, 1, 4, 3)

    v = torch.tensor(
        [[1, 0, 0],
         [10, 0, 0],
         [100, 5, 0],
         [1000, 6, 0]],
        dtype=torch.float32).view(1, 1, 4, 3)

    q = torch.tensor(
        [[0, 0, 10]],
        dtype=torch.float32).view(1, 1, 1, 3)

    out, attn = mha.scaled_dot_product_attention(q, k, v)
    print('out=', out)
    print('attn=', attn)


########################################################################################################################
def main_stupid2():
    """This time, the dataset. Cannot understand the new torchtext!"""
    dset_train, dset_val = torchtext.datasets.IMDB(root='/home/seymour/data/IMDB')
    print('dset_train', len(dset_train))
    print('dset_val', len(dset_val))

    batch_size = 164
    loader_train = torch.utils.data.DataLoader(dset_train, batch_size=batch_size)
    loader_val = torch.utils.data.DataLoader(dset_val, batch_size=batch_size)
    print('loader_train', len(loader_train))
    print('loader_val', len(loader_val))
    x, y = next(iter(loader_val))
    print('x', type(x), len(x))
    print('y', type(y), len(y))
    x0, y0 = x[0], y[0]
    print('x0', type(x0), x0)
    print('y0', type(y0), y0)
    # for x, y in loader_train:
    #     print('x=', x)


########################################################################################################################
def main_stupid3():
    """We use legacy stuff, like in the notebook"""
    # dataset + loaders
    import torchtext.legacy.data as data
    import torchtext.legacy.datasets as datasets
    max_len = 200
    text = data.Field(sequential=True, fix_length=max_len, batch_first=True, lower=True, dtype=torch.long)
    label = data.LabelField(sequential=False, dtype=torch.long)
    # datasets.IMDB.download('/home/seymour/data')
    ds_train, ds_test = datasets.IMDB.splits(text, label, path='/home/seymour/data/IMDB/aclImdb')
    # print('ds_train', len(ds_train))
    # print('ds_test', len(ds_test))
    print('train.fields :', ds_train.fields)
    ds_train, ds_valid = ds_train.split(0.9)
    print('train : ', len(ds_train))
    print('valid : ', len(ds_valid))
    print('test : ', len(ds_test))
    num_words = 50000
    text.build_vocab(ds_train, max_size=num_words)
    label.build_vocab(ds_train)
    vocab = text.vocab
    batch_size = 164
    train_loader, valid_loader, test_loader = data.BucketIterator.splits((ds_train, ds_valid, ds_test),
                                                                         batch_size=batch_size,
                                                                         sort_key=lambda x: len(x.text), repeat=False)
    # model etc
    # model = MultiHeadAttention(d_model=32, num_heads=2)
    # model = Embeddings(d_model=32, vocab_size=50002, max_position_embeddings=10000, p=0.1)
    # model = Encoder(num_layers=1, d_model=32, num_heads=2, ff_hidden_dim=128, input_vocab_size=50002,
    #                 max_position_embeddings=10000)
    model = TransformerClassifier()
    model.to(device=DEVICE)

    if False:
        batch = next(iter(train_loader))
        print('batch:', type(batch), len(batch))
        x = batch.text
        y = batch.label
        print_it(x, 'x')  # [164, 200]
        print_it(y, 'y')  # [164]
        print(y)
        print(x[0])

    if True:
        batch = next(iter(train_loader))
        x = batch.text.to(DEVICE)
        y = batch.label.to(DEVICE)
        print_it(x, 'x')  # [164, 200]
        print_it(y, 'y')  # [164]
        out = model(x)
        print_it(out, 'out')


########################################################################################################################
class Trainer:
    def __init__(self):
        # dataset + loaders
        import torchtext.legacy.data as data
        import torchtext.legacy.datasets as datasets
        max_len = 200
        text = data.Field(sequential=True, fix_length=max_len, batch_first=True, lower=True, dtype=torch.long)
        label = data.LabelField(sequential=False, dtype=torch.long)
        ds_train, ds_test = datasets.IMDB.splits(text, label, path='/home/seymour/data/IMDB/aclImdb')
        print('train.fields :', ds_train.fields)
        ds_train, ds_valid = ds_train.split(0.9)
        print('train : ', len(ds_train))
        print('valid : ', len(ds_valid))
        print('test : ', len(ds_test))
        num_words = 50000
        text.build_vocab(ds_train, max_size=num_words)
        label.build_vocab(ds_train)
        vocab = text.vocab
        batch_size = 164
        self.train_loader, self.valid_loader, self.test_loader = data.BucketIterator.splits(
            (ds_train, ds_valid, ds_test),
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False)
        print('train_loader : ', len(self.train_loader))
        print('valid_loader : ', len(self.valid_loader))
        print('test_loader : ', len(self.test_loader))
        # model etc
        self.model = TransformerClassifier().to(device=DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def val(self, loader, name: str = 'val'):
        self.model.eval()
        loss_val = 0.0
        acc_val = 0.0
        for batch in loader:
            x = batch.text.to(DEVICE)
            y = batch.label.to(DEVICE)
            with torch.no_grad():
                out = self.model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss_val += loss.item()
            acc_val += (out.argmax(1) == y).cpu().numpy().mean()
        loss_val /= len(loader)
        acc_val /= len(loader)
        print(f'{name} loss={loss_val}, acc={acc_val}')

    def train(self):
        epochs = 10
        for epoch in range(epochs):
            self.model.train()
            loss_train = 0.0
            acc_train = 0.0
            for batch in tqdm.tqdm(self.train_loader):
                x = batch.text.to(DEVICE)
                y = batch.label.to(DEVICE)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward()
                loss_train += loss.item()
                acc_train += (out.argmax(1) == y).cpu().numpy().mean()
                self.optimizer.step()
            loss_train /= len(self.train_loader)
            acc_train /= len(self.train_loader)
            print(f'epoch {epoch} : loss_train={loss_train}, acc_train={acc_train}')
            self.val(self.valid_loader)
        self.val(self.test_loader, 'TEST')


########################################################################################################################
def main():
    trainer = Trainer()
    trainer.train()


########################################################################################################################
if __name__ == '__main__':
    main()
    # main_stupid1()
