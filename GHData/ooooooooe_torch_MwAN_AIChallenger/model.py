# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class MwAN(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_size, encoder_size, drop_out=0.2):
        super(MwAN, self).__init__()
        self.drop_out=drop_out
        #self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)
        self.q_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.p_encoder = nn.GRU(input_size=embedding_size + 5, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.a_encoder = nn.GRU(input_size=embedding_size, hidden_size=int(embedding_size / 2), batch_first=True,
                                bidirectional=True)
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        self.att_attention = nn.Linear(2 * encoder_size, 1, bias=True)

        # Concat Attention
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)
        # dot attention between query
        self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)

        # qanet
        self.Wqa1 = nn.Linear(2 * encoder_size, 1, bias=False)
        self.Wqa2 = nn.Linear(2 * encoder_size, 1, bias=False)
        self.Wqa3 = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)

        self.gru_agg = nn.GRU(2 * encoder_size, encoder_size, batch_first=True, bidirectional=True)
        """
        prediction layer
        """
        self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.initiation()

    def initiation(self):
        # initrange = 0.1
        # nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def feature_extraction(self, passage, query, answer):
        #print(answer)
        batch_size = passage.shape[0]
        p_length = passage.shape[1]
        q_length = query.shape[1]
        qe_match = np.empty((batch_size, p_length))
        qe_hard_match = np.empty((batch_size, p_length))
        a1_match = np.empty((batch_size, p_length))
        a2_match = np.empty((batch_size, p_length))
        a3_match = np.empty((batch_size, p_length))
        passage_list = passage.cpu().numpy().tolist()
        query_list = query.cpu().numpy().tolist()
        answer_list = answer.cpu().numpy().tolist()
        for i in range(batch_size):
            a1_set = set(answer_list[i][0])
            a2_set = set(answer_list[i][1])
            a3_set = set(answer_list[i][2])
            q_set = set(query_list[i])
            for pos in range(p_length):
                #print(passage_np[i])
                #print(max(0, pos - q_length / 2), min(p_length - 1, pos + q_length / 2))
                p_set = set(passage_list[i][max(0, int(pos - q_length / 2)): min(p_length - 1, int(pos + q_length / 2))])
                # soft match Q and E
                qe_match[i][pos] = float(len(p_set & q_set)) / float(len(p_set | q_set))
                # hard match Q and E
                if passage_list[i][pos] in q_set:
                    qe_hard_match[i][pos] = 1
                else:
                    qe_hard_match[i][pos] = 0
                if passage_list[i][pos] in a1_set:
                    a1_match[i][pos] = 1
                else:
                    a1_match[i][pos] = 0
                if passage_list[i][pos] in a2_set:
                    a2_match[i][pos] = 1
                else:
                    a2_match[i][pos] = 0
                if passage_list[i][pos] in a3_set:
                    a3_match[i][pos] = 1
                else:
                    a3_match[i][pos] = 0
        return torch.FloatTensor(qe_match).cuda(), torch.FloatTensor(qe_hard_match).cuda(), torch.FloatTensor(a1_match).cuda(), torch.FloatTensor(a2_match).cuda(), torch.FloatTensor(a3_match).cuda()

    def forward(self, inputs):
        [query, passage, answer, is_train] = inputs
        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        a_embeddings = self.embedding(answer)
        a_embedding, _ = self.a_encoder(a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3)))
        a_score = F.softmax(self.a_attention(a_embedding), 1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()
        a_embedding = a_output.view(a_embeddings.size(0), 3, -1)
        qe_match, qe_hard_match, a1_match, a2_match, a3_match = self.feature_extraction(passage, query, answer)
        #.unsqueeze(2)
        qe_match = qe_match.unsqueeze(2)
        qe_hard_match = qe_hard_match.unsqueeze(2)
        a1_match = a1_match.unsqueeze(2)
        a2_match = a2_match.unsqueeze(2)
        a3_match = a3_match.unsqueeze(2)
        p_embedding = torch.cat([p_embedding, qe_match, qe_hard_match, a1_match, a2_match, a3_match], 2)

        hp, _ = self.p_encoder(p_embedding)
        hp=F.dropout(hp,self.drop_out)
        hq, _ = self.q_encoder(q_embedding)
        hq=F.dropout(hq,self.drop_out)
        a_hq = torch.tanh(self.att_attention(hq))

        # concat
        _s1 = self.Wc1(hp).unsqueeze(1)   # [batch, 1, passage_len, encoder_size]
        _s2 = self.Wc2(hq).unsqueeze(2)   # [batch, query_len, 1, encoder_size]
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()   # [batch, query_len, passage_len]
        ait = F.softmax(sjt, 2)
        qtc = ait.bmm(hp)  # [batch, query_len, encoder_size]
        a_qtc = torch.tanh(self.att_attention(qtc))#.repeat(1, 1, encoder_size)
        #qtc = qtc.mul(a_qtc)

        # bilinear
        _s1 = self.Wb(hp).transpose(2, 1)  # [batch, encoder_size, passage_len]
        sjt = hq.bmm(_s1)   # [batch, query_len, passage_len]
        ait = F.softmax(sjt, 2)
        qtb = ait.bmm(hp)
        a_qtb = torch.tanh(self.att_attention(qtb))#.repeat(1, 1, encoder_size)
        #qtb = qtb.mul(a_qtb)

        # dot
        _s1 = hp.unsqueeze(1)
        _s2 = hq.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtd = ait.bmm(hp)
        a_qtd = torch.tanh(self.att_attention(qtd))#.repeat(1, 1, encoder_size)
        #qtd = qtd.mul(a_qtd)

        # minus
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtm = ait.bmm(hp)
        a_qtm = torch.tanh(self.att_attention(qtm))#.repeat(1, 1, encoder_size)
        #qtm = qtm.mul(a_qtm)

        # dot, self-attention of query
        _s1 = hq.unsqueeze(1)           # [batch, 1, query_len, encoder_size]
        _s2 = hq.unsqueeze(2)           # [batch, query_len, 1, encoder_size]
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qts = ait.bmm(hq)
        a_qts = torch.tanh(self.att_attention(qts))#.repeat(1, 1, encoder_size)
        #qts = qts.mul(a_qts)

        # qanet
        q_len = query.shape[1]
        p_len = passage.shape[1]
        subres0 = self.Wqa1(hp).repeat(1,1,q_len).transpose(2,1)   # [batch, query_len, passage_len]
        subres1 = self.Wqa2(hq).repeat(1,1,p_len)                  # [batch, query_len, passage_len]
        subres3 = self.Wqa3(hq).bmm(hp.transpose(2,1))             # [batch, query_len, passage_len]
        sim_mat = subres0 + subres1 + subres3                      # [batch, query_len, passage_len]]
        S_ = F.softmax(sim_mat, 2)       # [batch, query_len, passage_len]]
        p2q = S_.bmm(hp)                 # [batch, query_len, encoder_size]
        S_T = F.softmax(sim_mat, 1).transpose(2,1)  # [batch, passage_len, query_len]
        q2p = S_.bmm(S_T).bmm(hq)
        a_p2q = torch.tanh(self.att_attention(q2p))#.repeat(1, 1, encoder_size)
        #p2q = p2q.mul(a_p2q)
        a_q2p = torch.tanh(self.att_attention(q2p))#.repeat(1, 1, encoder_size)
        #q2p = q2p.mul(a_q2p)

        # aggregation
        #hq = hq.unsqueeze(1)
        #qts = qts.unsqueeze(1)
        #qtc = qtc.unsqueeze(1)
        #qtd = qtd.unsqueeze(1)
        #qtb = qtb.unsqueeze(1)
        #qtm = qtm.unsqueeze(1)
        #p2q = p2q.unsqueeze(1)
        #q2p = q2p.unsqueeze(1)
        a_aggregation = torch.cat([a_hq, a_qts, a_qtc, a_qtd, a_qtb, a_qtm, a_p2q, a_q2p], 2)
        soft_aggregation = F.softmax(a_aggregation, 2)
        #print (torch.split(soft_aggregation, 8, 2))
        a_hq, a_qts, a_qtc, a_qtd, a_qtb, a_qtm, a_p2q, a_q2p = torch.split(soft_aggregation, 1, 2)
        hq = hq.mul(a_hq.repeat(1, 1, hp.shape[2]))
        qts = qts.mul(a_qts.repeat(1, 1, qts.shape[2]))
        qtc = qtc.mul(a_qtc.repeat(1, 1, qtc.shape[2]))
        qtd = qtd.mul(a_qtd.repeat(1, 1, qtd.shape[2]))
        qtb = qtb.mul(a_qtb.repeat(1, 1, qtb.shape[2]))
        qtm = qtm.mul(a_qtm.repeat(1, 1, qtm.shape[2]))
        p2q = p2q.mul(a_p2q.repeat(1, 1, p2q.shape[2]))
        q2p = q2p.mul(a_q2p.repeat(1, 1, q2p.shape[2]))

        #aggregation = torch.cat([hq, qts, qtc, qtd, qtb, qtm, p2q, q2p], 2)
        aggregation = torch.sum(torch.cat([hq.unsqueeze(0), qts.unsqueeze(0), qtc.unsqueeze(0), qtd.unsqueeze(0), qtb.unsqueeze(0), qtm.unsqueeze(0), p2q.unsqueeze(0), q2p.unsqueeze(0)], 0), 0).squeeze(0)
        aggregation_representation, _ = self.gru_agg(aggregation)

        # predition layer
        sj = self.vq(torch.tanh(self.Wq(hq))).transpose(2, 1)
        rq = F.softmax(sj, 2).bmm(hq)
        sj = F.softmax(self.vp(self.Wp1(aggregation_representation) + self.Wp2(rq)).transpose(2, 1), 2)
        rp = sj.bmm(aggregation_representation)
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)),self.drop_out)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
        if not is_train:
            return score.argmax(1)
        loss = -torch.log(score[:, 0]).mean()
        pred = score.argmax(1)
        return loss, pred


