import torch
from Config import config
import random
from DataLoader import loader


class Encoder(torch.nn.Module):
    """
    Encoder
    用于将输入的句子编码为上下文向量 enc_outputs，以及获得 Decoder 的输入 hidden
    输入: (batch, sequence), 元素的值域为enc_vocab_size
    输出: enc_outputs: (batch, sequence, 2 * enc_hidden) hidden: (batch, dec_hidden)
    """
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.enc_vocab_size, config.enc_embed_size)
        self.rnn = torch.nn.GRU(config.enc_embed_size, config.enc_hidden_size, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(config.enc_hidden_size * 2, config.dec_hidden_size)
        self.dropout = torch.nn.Dropout(config.enc_dropout)

    def forward(self, enc_inputs):
        embed = self.dropout(self.embedding(enc_inputs))
        outputs, hidden = self.rnn(embed)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(torch.nn.Module):
    """
    Attention
    使用 dec_hidden 对 enc_outputs 进行 attention，获得每一个 enc_output 的权重
    输入: encode_outputs: (batch, sequence, 2 * enc_hidden), decode_hidden: (batch, dec_hidden)
    输出: att_weights: (batch, sequence)
    """
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Linear(config.enc_hidden_size * 2 + config.dec_hidden_size, config.attn_dim)

    def forward(self, encode_outputs, decode_hidden):
        seq_len = encode_outputs.shape[1]
        repeated_decode_hidden = decode_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        weights = torch.tanh(self.attention(torch.cat((encode_outputs, repeated_decode_hidden), dim=2)))
        weights = torch.sum(weights, dim=2)
        return torch.nn.functional.softmax(weights, dim=1)


class Decoder(torch.nn.Module):
    """
    Decoder
    一次解码序列中的一个位置，给出词典中的可能性(softmax), 以及下一轮解码的hidden
    输入: dec_input: (batch, word_idx) enc_outputs: (batch, sequence, 2 * enc_hidden) dec_hidden: (batch, dec_hidden)
    输出: output: (batch, dec_vocab_size) hidden: (batch, dec_hidden)
    """
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(config.dec_vocab_size, config.dec_embed_size)
        self.rnn = torch.nn.GRU(config.enc_hidden_size * 2 + config.dec_embed_size, config.dec_hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(config.enc_hidden_size * 2 + config.dec_hidden_size + config.dec_embed_size,
                                  config.dec_vocab_size)
        self.dropout = torch.nn.Dropout(config.dec_dropout)
        self.att = Attention()

    @staticmethod
    def _get_weighted_outputs(enc_outputs, weights):
        weights = weights.unsqueeze(1)
        weighted_encoder_rep = torch.bmm(weights, enc_outputs)
        return weighted_encoder_rep

    def forward(self, enc_outputs, dec_input, dec_hidden):
        dec_input = dec_input.unsqueeze(1)
        embed = self.dropout(self.embed(dec_input))
        att_weights = self.att(enc_outputs, dec_hidden)
        att_enc_outs = self._get_weighted_outputs(enc_outputs, att_weights)
        rnn_input = torch.cat((att_enc_outs, embed), dim=2)
        output, hidden = self.rnn(rnn_input, dec_hidden.unsqueeze(0))
        output = self.fc(torch.cat((att_enc_outs, output, embed), dim=2))
        return output.squeeze(1), hidden.squeeze(1)


class Seq2Seq(torch.nn.Module):
    """
    Seq2Seq 模型
    输入: (batch, sequence, enc_vocab_size)
    输出: (sequence, batch, dec_vocab_size)
    对 dic_size 维度取 max 即得到预测结果（词典中的索引）
    注意：需要作padding, 并且输出 batch 在第二维
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt, teacher_force_ratio=0.5):
        length = tgt.shape[1]
        enc_outputs, enc_hidden = self.encoder(src)
        outputs = torch.zeros(length, config.batch_size, config.dec_vocab_size).to(config.device)
        output = tgt[:, 0]
        for i in range(1, length):
            output, hidden = self.decoder(enc_outputs, output, enc_hidden)
            outputs[i] = output
            teacher_force = random.random() < teacher_force_ratio
            top1 = output.max(1)[1]
            output = (tgt[:, i] if teacher_force else top1)
        return outputs


model = Seq2Seq().to(config.device)
# # 训练示例
# optimizer = torch.optim.Adam(lr=config.lr, params=model.parameters())
# loss_fn = torch.nn.CrossEntropyLoss()
# for tensor1, tensor2 in loader.run():
#     outputs = model(tensor1, tensor2)
#     loss = 0
#     temp = tensor2.permute(1, 0)
#     for output, y in zip(outputs, temp):
#         loss += loss_fn(output, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(loss)
