import torch
from torch import nn
from torch import cuda
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models.interfaces import TranslationModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader, load_wn18rr

from tqdm.autonotebook import tqdm

# Load dataset
kg_train, kg_val, kg_test = load_wn18rr()

# Define some hyper-parameters for training
emb_dim = 100
lr = 0.0004
n_epochs = 10
b_size = 32768
margin = 0.5

class BaseTransD(TranslationModel):
    def __init__(self, num_entities, num_relations, dim=100):
        super(BaseTransD, self).__init__(num_entities, num_relations, dissimilarity_type='L2')
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)
        self.ent_vect = nn.Embedding(num_entities, self.dim)
        self.rel_vect = nn.Embedding(num_relations, self.dim)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

        self.normalize_parameters()
        self.evaluated_projections = False
        self.projected_entities = nn.Parameter(torch.empty(size=(num_relations, num_entities, self.dim)), requires_grad=False)

    def normalize_parameters(self):
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_vect.weight.data = F.normalize(self.ent_vect.weight.data, p=2, dim=1)
        self.rel_vect.weight.data = F.normalize(self.rel_vect.weight.data, p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_embeddings.weight.data, self.ent_vect.weight.data, self.rel_vect.weight.data

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        if not self.evaluated_projections:
            self.lp_evaluate_projections()

        r = self.rel_embeddings(r_idx)
        proj_h = self.projected_entities[r_idx, h_idx]
        proj_t = self.projected_entities[r_idx, t_idx]
        proj_candidates = self.projected_entities[r_idx]
        return proj_h, proj_t, proj_candidates, r

    def lp_evaluate_projections(self):
        if self.evaluated_projections:
            return
        for i in tqdm(range(self.num_entities), unit='entities', desc='Projecting entities'):
            rel_vects = self.rel_vect.weight.data
            ent = self.ent_embeddings.weight[i]
            ent_vect = self.ent_vect.weight[i]
            sc_prod = (ent_vect * ent).sum(dim=0)
            proj_e = sc_prod * rel_vects + ent[:self.dim].view(1, -1)
            self.projected_entities[:, i, :] = proj_e
            del proj_e
        self.evaluated_projections = True

    def forward(self, h, t, nh, nt, r):
        return self.scoring_function(h, t, r), self.scoring_function(nh, nt, r)

    def project(self, ent, ent_vect, rel_vect):
        proj_e = (rel_vect * (ent * ent_vect).sum(dim=1).view(ent.shape[0], 1))
        return proj_e + ent[:, :self.dim]

    @staticmethod
    def l2_dissimilarity(a, b):
        assert len(a.shape) == len(b.shape)
        return (a-b).norm(p=2, dim=-1)**2


class TransD(BaseTransD):
    def scoring_function(self, h_idx, t_idx, r_idx):
        self.evaluated_projections = False
        h = F.normalize(self.ent_embeddings(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t_idx), p=2, dim=1)
        r = F.normalize(self.rel_embeddings(r_idx), p=2, dim=1)

        h_proj = F.normalize(self.ent_vect(h_idx), p=2, dim=1)
        t_proj = F.normalize(self.ent_vect(t_idx), p=2, dim=1)
        r_proj = F.normalize(self.rel_vect(r_idx), p=2, dim=1)

        scores = -torch.norm(self.project(h, h_proj, r_proj) + r - self.project(t, t_proj, r_proj), 2, -1)
        return scores

class MarginLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def forward(self, positive_scores, negative_scores):
        return self.loss(positive_scores, negative_scores, target=torch.ones_like(positive_scores))


# Define model
model = TransD(kg_train.n_ent, kg_train.n_rel, dim=64)

# Define criterion for training model
criterion = MarginLoss(margin=0.5)


# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()

# Define the torch optimizer to be used
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Define negative sampler
sampler = BernoulliNegativeSampler(kg_train)

# Define Dataloader
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')

# Training loop
iterator = tqdm(range(n_epochs), unit='epoch')
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)

        optimizer.zero_grad()

        # forward + backward + optimize
        pos, neg = model(h, t, n_h, n_t, r)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    iterator.set_description('Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(dataloader)))

model.normalize_parameters()


# Define evaluator
evaluator = LinkPredictionEvaluator(model, kg_test)

# Run evaluator
evaluator.evaluate(b_size=128)

# Show results
print("----------------Overall Results----------------")
print('Hit@10: {:.4f}'.format(evaluator.hit_at_k(k=10)[0]))
print('Hit@3: {:.4f}'.format(evaluator.hit_at_k(k=3)[0]))
print('Hit@1: {:.4f}'.format(evaluator.hit_at_k(k=1)[0]))
print('Mean Rank: {:.4f}'.format(evaluator.mean_rank()[0]))
print('Mean Reciprocal Rank : {:.4f}'.format(evaluator.mrr()[0]))