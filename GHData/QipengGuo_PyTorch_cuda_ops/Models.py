from ExpHead import *
from bmv import *
from torch.autograd import Function
import os 

class BatchedMV3(Function):
    @staticmethod
    def forward(ctx, A, B0, B1, B2, B3):
        A = A.contiguous()
        B0 = B0.contiguous()
        B1 = B1.contiguous()
        B2 = B2.contiguous()
        B3 = B3.contiguous()
        y = bmv4_forward1(A,B0,B1,B2,B3)
        ctx.save_for_backward(A, B0,B1,B2,B3)
        return y

    @staticmethod
    def backward(ctx, dy):
        A, B0, B1, B2, B3 = ctx.saved_tensors
        dB = A[:,None,:] * dy[:,:,None]
        dA = bmv4_forward2(dy, B0, B1, B2, B3)
        return dA, dB[:,0], dB[:,1], dB[:,2], dB[:,3]

class BatchedMV4(Function):
    @staticmethod
    def forward(ctx, A, B0, B1, B2, B3):
        A = A.contiguous()
        B0 = B0.contiguous()
        B1 = B1.contiguous()
        B2 = B2.contiguous()
        B3 = B3.contiguous()
        y = bmv4_forward2(A,B0,B1,B2,B3)
        ctx.save_for_backward(A, B0,B1,B2,B3)
        return y

    @staticmethod
    def backward(ctx, dy):
        A, B0, B1, B2, B3 = ctx.saved_tensors
        dB = A[:,:,None] * dy[:,None,:]
        dA = bmv4_forward1(dy, B0, B1, B2, B3)
        return dA, dB[:,0], dB[:,1], dB[:,2], dB[:,3]

class BatchedMV1(Function):
    @staticmethod
    def forward(ctx, A, B):
        A = A.contiguous()
        B = B.contiguous()
        y = bmv_forward1(A,B)
        ctx.save_for_backward(A, B)
        return y

    @staticmethod
    def backward(ctx, dy):
        A, B = ctx.saved_tensors
        dB = A[:,:,None] * dy[:,None,:]
        dA = bmv_forward2(dy, B)
        return dA, dB

class BatchedMV2(Function):
    @staticmethod
    def forward(ctx, A, B):
        A = A.contiguous()
        B = B.contiguous()
        y = bmv_forward2(A,B)
        ctx.save_for_backward(A, B)
        return y

    @staticmethod
    def backward(ctx, dy):
        A, B = ctx.saved_tensors
        dB = A[:,None,:] * dy[:,:,None]
        dA = bmv_forward1(dy, B)
        return dA, dB

cbmv1 = BatchedMV1.apply
cbmv2 = BatchedMV2.apply
cbmv3 = BatchedMV3.apply
cbmv4 = BatchedMV4.apply

class MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10):
        super(MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Linear(nhid, nhead*head_dim)
        self.WK = nn.Linear(nhid, nhead*head_dim)
        self.WV = nn.Linear(nhid, nhead*head_dim)
        self.WO = nn.Linear(nhead*head_dim, nhid)

        self.drop = nn.Dropout(0.1)

        print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    # grad q, k, v, ax
    def forward(self, x, ax=None):
        # x: B, L, H; ax: B, L, X, H
        torch.cuda.synchronize()
        timer.be('MSA1')
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, L, H = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        tk, tv = [], []
        if ax is not None:
            ak, av = self.WK(ax), self.WV(ax)
            tk.append(ak[:,None,:].expand(B,L,nhead*head_dim).contiguous().view(B*L*nhead, head_dim))
            tv.append(av[:,None,:].expand(B,L,nhead*head_dim).contiguous().view(B*L*nhead, head_dim))
        #q, k, v = self.WQ(x), self.WK(x), self.WV(x) 
        torch.cuda.synchronize()
        timer.be('pad')
        pad_k = F.pad(k, (0,0,unfold_size//2,unfold_size//2))
        pad_v = F.pad(k, (0,0,unfold_size//2,unfold_size//2))
        for t in range(unfold_size):
            tk.append(pad_k[:,t:L+t].contiguous().view(B*L*nhead, head_dim))
            tv.append(pad_v[:,t:L+t].contiguous().view(B*L*nhead, head_dim))
        q = q.view(B*L*nhead, head_dim)
        if ax is not None:
            unfold_size += 1
        torch.cuda.synchronize()
        timer.en('pad')

        alphas = self.drop(F.softmax(cbmv3(q,tk[0], tk[1], tk[2], tk[3])/NP.sqrt(head_dim), 1)) 
        att = cbmv4(alphas,tv[0], tv[1], tv[2], tv[3]).view(B,L,nhead*head_dim)
        #alphas = self.drop(F.softmax(cbmv1(q,k)/NP.sqrt(head_dim), 1)) 
        #att = cbmv2(alphas,v).view(B,L,nhead*head_dim)

        ret = self.WO(att)
    
        torch.cuda.synchronize()
        timer.en('MSA1')
        return ret 


class MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(MSA2, self).__init__()
        self.WQ = nn.Linear(nhid, nhead*head_dim)
        self.WK = nn.Linear(nhid, nhead*head_dim)
        self.WV = nn.Linear(nhid, nhead*head_dim)
        self.WO = nn.Linear(nhead*head_dim, nhid)

        self.drop = nn.Dropout(0.1)

        print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H ; y: B, L, H ; mask: B, L
        torch.cuda.synchronize()
        timer.be('MSA2')
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, L, H = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B*nhead, head_dim)
        k = k.view(B, L, nhead, head_dim).permute(0,2,1,3).contiguous().view(B*nhead, L, head_dim)
        v = v.view(B, L, nhead, head_dim).permute(0,2,1,3).contiguous().view(B*nhead, L, head_dim)
        pre_a = cbmv2(q, k)/NP.sqrt(head_dim)
#        if mask is not None:
#            pre_a = pre_a.masked_fill(mask[:,None,:].expand(B,nhead,L).contiguous().view(B*nhead,L), -1e10)
        alphas = self.drop(F.softmax(pre_a, 1)) # B*N, L
        att = cbmv1(alphas, v).view(B, nhead*head_dim)
        att = self.WO(att)
        timer.en('MSA2')
        torch.cuda.synchronize()
        return att

class StarTrans(nn.Module):
    def __init__(self, nhid, V_DIM, L_DIM, embs, args):
        super(StarTrans, self).__init__()
        self.iters = args.iters
        nemb = 400 if args.use_char else 300

        self.emb_fc = nn.Linear(nemb, nhid)
        self.emb = nn.Embedding(V_DIM, nemb)
        self.emb.weight.data.copy_(embs)
        if args.free_emb:
            self.emb.weight.requires_grad = True
        else:
            self.emb.weight.requires_grad = False

        self.norm_1 = nn.ModuleList([ nn.LayerNorm(nhid) for _ in range(args.iters) ])
        self.ring_att = nn.ModuleList([ MSA1(nhid, nhead=args.num_head, head_dim=args.head_dim) for _ in range(args.iters) ])
        self.star_att = nn.ModuleList([ MSA2(nhid, nhead=args.num_head, head_dim=args.head_dim) for _ in range(args.iters) ])

        self.pos_emb = nn.Embedding(1000, nhid)
        self.emb_drop = nn.Dropout(args.emb_drop)
        self.nhid, self.args = nhid, args

    def forward(self, data):
        B,L = data.size()
        H = self.nhid
        mask = data==1
        smask = torch.cat([torch.zeros(B, 1).byte().cuda(), mask], 1)
        embs = self.emb(data)
        P = self.pos_emb(torch.arange(L).long().cuda().view(1,L))
        embs = self.emb_fc(self.emb_drop(embs)) + P 
    
        nodes = embs
        relay = embs.mean(1)
        for i in range(self.iters):
            #ax = torch.stack([embs, relay[:,None,:].expand(B,L,H)], 2)
            #ax = relay[:,None,:].expand(B,L,H)
            ax = relay
            nodes = nodes + F.leaky_relu(self.ring_att[i](self.norm_1[i](nodes), ax = ax))
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay[:,None,:], nodes], 1), smask))

            nodes = nodes.masked_fill_(mask[:,:,None], 0)

        rep = 0.5 * nodes.max(1)[0] + 0.5 * relay.view(B, H)
        return rep

class Cls(nn.Module):
    def __init__(self, hid_dim, L_DIM, args):
        super(Cls, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, getattr(args, 'cls_dim', 600)),
            nn.LeakyReLU(),
            nn.Dropout(getattr(args, 'cls_drop', 0.5)),
            nn.Linear(getattr(args, 'cls_dim', 600), L_DIM)
        )

    def forward(self, x):
        h = self.fc(x)
        return F.log_softmax(h, -1)

class NLICls(nn.Module):
    def __init__(self, hid_dim, L_DIM, args):
        super(NLICls, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(getattr(args, 'cls_drop', 0.5)),
            nn.Linear(hid_dim*4, getattr(args, 'cls_hid', 600)),  #4
            nn.LeakyReLU(),
            nn.Dropout(getattr(args, 'cls_drop', 0.5)),
            nn.Linear(getattr(args, 'cls_hid', 600), L_DIM)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2, torch.abs(x1-x2), x1*x2], 1)
        #x = torch.cat([x1, x2], 1)
        h = self.fc(x)
        return F.log_softmax(h, 1)

class ModelZoo(nn.Module):
    def __init__(self, args, V_DIM, L_DIM, embs):
        super(ModelZoo, self).__init__()
        if args.mode == 'StarTrans':
            self.enc = StarTrans(args.hid_dim, V_DIM,  L_DIM, embs, args)

        if args.dataset=='SNLI':
            self.cls = NLICls(args.hid_dim, L_DIM, args)
        else:
            self.cls = Cls(args.hid_dim, L_DIM, args)
        self.args, self.L_DIM = args, L_DIM

    def forward(self, Dx):
        if self.args.dataset=='SNLI':
            if self.args.cross_input:
                rep1, rep2 = self.enc(Dx['premise'], Dx['hypothesis'])
            else:
                rep1 = self.enc(Dx['premise'])
                rep2 = self.enc(Dx['hypothesis'])
            pred = self.cls(rep1, rep2)

        if self.args.dataset in ['ONTO_POS', 'ONTO_NER', 'PTB', 'CONLL_NER', 'SST-2', 'SST-5'] or 'MTL' in self.args.dataset:
            if Dx['text'].size(1)>512: #and self.args.mode == 'Trans':
                Dx['text'] = Dx['text'][:,:512]
                #print('Too Long')
            rep = self.enc(Dx['text'])
            pred = self.cls(rep)

        #print(pred.shape, Dx['label'].shape)
        loss = F.nll_loss(pred, Dx['label'])

        return {'pred':pred, 'loss':loss}
