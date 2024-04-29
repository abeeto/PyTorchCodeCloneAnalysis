import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        mzy = e1_embedded*rel_embedded
        yzm0 = self.emb_e.weight
        yzm = self.emb_e.weight.transpose(1,0)
        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = torch.sigmoid(pred)

        return pred



class ED_SimplE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ED_SimplE, self).__init__()
        self.emb_e_head_re = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_head_im = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)

        self.emb_e_tail_re = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_tail_im = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)

        self.emb_rel_re = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_im = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)

        self.emb_rel_inv_re = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_inv_im = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_head_re.weight.data)
        xavier_normal_(self.emb_e_tail_re.weight.data)
        xavier_normal_(self.emb_rel_re.weight.data)
        xavier_normal_(self.emb_rel_inv_re.weight.data)

        xavier_normal_(self.emb_e_head_im.weight.data)
        xavier_normal_(self.emb_e_tail_im.weight.data)
        xavier_normal_(self.emb_rel_im.weight.data)
        xavier_normal_(self.emb_rel_inv_im.weight.data)

    def forward(self, e1, rel):
        e1_embedded_head_re = self.emb_e_head_re(e1)
        e1_embedded_head_im = self.emb_e_head_im(e1)

        e1_embedded_tail_re = self.emb_e_tail_re(e1)
        e1_embedded_tail_im = self.emb_e_tail_im(e1)

        rel_embedded_re = self.emb_rel_re(rel)
        rel_embedded_im = self.emb_rel_im(rel)

        rel_embedded_inv_re = self.emb_rel_inv_re(rel)
        rel_embedded_inv_im = self.emb_rel_inv_im(rel)

        e1_embedded_head_re = e1_embedded_head_re.squeeze()
        e1_embedded_head_im = e1_embedded_head_im.squeeze()

        e1_embedded_tail_re = e1_embedded_tail_re.squeeze()
        e1_embedded_tail_im = e1_embedded_tail_im.squeeze()

        rel_embedded_re = rel_embedded_re.squeeze()
        rel_embedded_im = rel_embedded_im.squeeze()

        rel_embedded_inv_re = rel_embedded_inv_re.squeeze()
        rel_embedded_inv_im = rel_embedded_inv_im.squeeze()

        e1_embedded_head_re = self.inp_drop(e1_embedded_head_re)
        e1_embedded_head_im = self.inp_drop(e1_embedded_head_im)
        e1_embedded_tail_re = self.inp_drop(e1_embedded_tail_re)
        e1_embedded_tail_im = self.inp_drop(e1_embedded_tail_im)

        rel_embedded_re = self.inp_drop(rel_embedded_re)
        rel_embedded_im = self.inp_drop(rel_embedded_im)
        rel_embedded_inv_re = self.inp_drop(rel_embedded_inv_re)
        rel_embedded_inv_im = self.inp_drop(rel_embedded_inv_im)


        # realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        # realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        # imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        # imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))

        dot1 = torch.mm(e1_embedded_head_re * rel_embedded_re, self.emb_e_tail_re.weight.transpose(1, 0))
        dot2 = torch.mm(e1_embedded_head_re * rel_embedded_im, self.emb_e_tail_im.weight.transpose(1, 0))
        dot3 = torch.mm(e1_embedded_head_im * rel_embedded_re, self.emb_e_tail_im.weight.transpose(1, 0))
        dot4 = torch.mm(e1_embedded_head_im * rel_embedded_im, self.emb_e_tail_re.weight.transpose(1, 0))
        pred1 = dot1 + dot2 + dot3 - dot4

        dot1_inv = torch.mm(e1_embedded_tail_re * rel_embedded_inv_re, self.emb_e_head_re.weight.transpose(1, 0))
        dot2_inv = torch.mm(e1_embedded_tail_re * rel_embedded_inv_im, self.emb_e_head_im.weight.transpose(1, 0))
        dot3_inv = torch.mm(e1_embedded_tail_im * rel_embedded_inv_re, self.emb_e_head_im.weight.transpose(1, 0))
        dot4_inv = torch.mm(e1_embedded_tail_im * rel_embedded_inv_im, self.emb_e_head_re.weight.transpose(1, 0))
        pred2 = dot1_inv + dot2_inv + dot3_inv - dot4_inv

        pred = (pred1 + pred2)/2.0
        # pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        # pred = (torch.mm(e1_embedded_head * rel_embedded, self.emb_e_tail.weight.transpose(1, 0)) + torch.mm(e1_embedded_tail * rel_embedded_inv, self.emb_e_head.weight.transpose(1, 0))) * 0.5
        # pred1 = torch.mm(e1_embedded_head * rel_embedded, self.emb_e_tail.weight.transpose(1, 0))
        # pred2 = torch.mm(e1_embedded_tail * rel_embedded_inv, self.emb_e_head.weight.transpose(1,0))
        pred = torch.sigmoid(pred)

        return pred



# Add your own model here
class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = torch.sigmoid(output)

        return prediction
