import torch.nn as nn

from utils

class Embedder(nn.Module):
    """
    単語idに対応する埋め込み表現を取得するモジュール
    """
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)
        # freeze -> does not change in backprop.

    def forward(self, x):
        x_embed = self.embeddings(x)

        return x_embed


class PositionalEncoder(nn.Module):
    """
    入力された単語の位置情報をEncoding
    """
    def __init__(self, d_model=300, max_seq_len=256):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model  # 単語ベクトルの次元数
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pos_enc = torch.zeros(max_seq_len, d_model).to(device)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.pos_enc = pos_enc.unsqeeze(0)
        self.pos_enc.requires_grad = False

    def forward(self, x):
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


class Attention(nn.Module):
    # single head attention
    def __init__(self, d_model=300):
        super(Attention, self).__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)
        normalized_weights = F.softmax(weights, dim=-1)

        output = torch.matmul(normalized_weights, v)
        output = self.out(output)

        return output, normalized_weights
