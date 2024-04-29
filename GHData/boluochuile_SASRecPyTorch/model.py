import numpy as np
import torch
from cluster.kmeans import kmeans

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()

        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.hidden_units = args.hidden_units
        self.batch_size = args.batch_size
        self.maxlen = args.maxlen
        self.num_interest = args.num_interest
        self.softmax = torch.nn.Softmax(1)
        # self.liner1 = torch.nn.Linear(self.hidden_units * 4)
        # self.a1 = torch.nn.Tanh

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        # positions = self.positional_encoding(log_seqs)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        # seqs += positions
        seqs = self.emb_dropout(seqs)

        # log_seqs等于0的变为True，其余的为False
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # ~ 取反
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        #         # sql_len
        # tl = seqs.shape[1] # time dim len for enforce causality
        # # ~torch.tril: 上三角为True(不包括对角线)，下三角为False(包括对角线)
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # (sql_len, batch_size, embedding_dim)
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs)
                                            # attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        # (b, seq_len, embedding_dim)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        # (b*num_interest, embedding_dim)
        user_eb = None
        for i in range(log_feats.shape[0]):
            # best_centers, best_distance
            best_centers, best_distance = kmeans(log_feats[i], self.num_interest, batch_size=self.maxlen, iter=5)
            if user_eb is None:
                user_eb = best_centers
            else:
                user_eb = torch.cat((user_eb, best_centers), 0)

        return user_eb.view(log_feats.shape[0], self.num_interest, self.hidden_units), log_feats

    def log2feats2(self, log_seqs):

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        # log_seqs等于0的变为True，其余的为False
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # ~ 取反
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        # (b, seq_len, embedding_dim)
        # log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        # (b*num_interest, embedding_dim)
        user_eb = None
        for i in range(seqs.shape[0]):
            # best_centers, best_distance
            best_centers, best_distance = kmeans(seqs[i], self.num_interest, batch_size=self.maxlen, iter=5)
            if user_eb is None:
                user_eb = best_centers
            else:
                user_eb = torch.cat((user_eb, best_centers), 0)

        return user_eb.view(seqs.shape[0], self.num_interest, self.hidden_units), seqs


    def forward(self, log_seqs): # for training

        user_eb, log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        # neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # neg_logits = (log_feats * neg_embs).sum(dim=-1)
        #
        # user_eb = user_eb.view(self.batch_size, self.num_interest, self.hidden_units)
        # pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # pos_embs = pos_embs[:, -1, :].unsqueeze(1).repeat(1, self.num_interest, 1)
        # pos_logits = (user_eb * pos_embs).sum(dim=-1)
        # user_eb = user_eb.view(-1, self.hidden_units)
        # h = torch.from_numpy(np.array([i * self.num_interest for i in range(self.batch_size)])).to(self.dev)
        # # (b, 1)
        # index = torch.argmax(pos_logits, -1)
        # user_eb = torch.index_select(user_eb, 0, index + h)
        #
        # temp = self.item_emb(torch.LongTensor(pos_seqs[:, -1]).to(self.dev))
        # user_eb = (user_eb * temp).sum(dim=-1)

        return user_eb

    def predict(self, log_seqs): # for inference
        user_eb, log_feats = self.log2feats(log_seqs)
        return user_eb

    def output_item(self):
        return self.item_emb

    # 找出与候选向量最相似的用户兴趣
    def output_user(self, log_seqs, item_list, hist_mask):
        # (b, num_interest, embedding_dim)
        user_eb, log_feats = self.log2feats(log_seqs)
        # (b, embedding_dim)
        item_list_emb = self.item_emb(torch.from_numpy(item_list).long().to(self.dev))

        # atten: (batch, num_interest, dim) * (batch, dim, 1) = (batch, num_interest, 1)
        atten = torch.matmul(user_eb, item_list_emb.view(item_list_emb.shape[0], self.hidden_units, 1))
        # (batch, num_interest)
        atten = self.softmax(torch.pow(atten.view(item_list_emb.shape[0], self.num_interest), 1))

        indices = (torch.argmax(atten, dim=1).int()
                   + (torch.range(0, item_list_emb.shape[0] - 1) * self.num_interest).to(self.dev)).long()
        # 找出与target item最相似的用户兴趣向量
        readout = torch.index_select(user_eb.view(-1, self.hidden_units), 0, indices)

        return readout



    def positional_encoding(self, seq_inputs):
        encoded_vec = [pos / np.power(10000.0, 2 * i / self.hidden_units)
                       for pos in range(seq_inputs.shape[-1]) for i in range(self.hidden_units)]
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        encoded_vec = torch.from_numpy(np.array(encoded_vec)).view(-1, self.hidden_units).float().to(self.dev)

        return encoded_vec