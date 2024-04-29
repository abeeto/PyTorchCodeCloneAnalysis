import pickle
import torch
import numpy as np
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from   torch.nn.init import xavier_normal_, xavier_uniform_, constant_, normal_, uniform_
from   torch.utils.data import Dataset, DataLoader
import logging
import argparse
import os
import psutil


def get_memory_info():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return np.round(info.uss / 1024**3, 1)


def get_logger(log_path="out.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(stream_handler)
    return logger

class EarlyStopper():
    def __init__(self, num_trials=10):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_result = np.inf
    def is_continuable(self, result):
        if  result < self.best_result:
            self.best_result = result
            self.trial_counter = 0
            print("Best result is {:.4f}.".format(result))
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def get_model(name, user_num, item_num, embedding_dim, hidden_dim=None, user_review_encoder=None, item_review_encoder=None):
    if name == "narre":
        model = NARRE(user_num + 3, item_num + 3, embedding_dim, hidden_dim, user_review_encoder, item_review_encoder)
        model.name = 'narre'
    elif name == "mf":
        model = NARRE(user_num + 3, item_num + 3, embedding_dim)
        model.name = "mf"
    else:
        raise ValueError("unknown model {}.".format(name))
    return model


class Dataset(Dataset):
    def __init__(self, data, u_text, i_text):
        self.data = data
        self.u_text = u_text
        self.i_text = i_text
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        uid, iid, reuid, reiid, target = self.data[i]
        u_batch = self.u_text[uid[0]]
        i_batch = self.i_text[iid[0]]
        return u_batch, i_batch, uid, iid, reuid, reiid, target

def get_dataset(name):
    print("Loading data {}...".format(name))
    data_base = "/code/r9/xai/NARRE/data/{}/".format(name)
    para_data  = data_base+'music.para'
    print("Start load para file ...")
    with open(para_data, 'rb') as f:
        para = pickle.load(f)
    print("done. {}G data loaded.".format(get_memory_info()))
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    user_vocab = para['user_vocab']
    item_vocab = para['item_vocab']
    train_length = para['train_length']
    test_length  = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']
    print('user_num', user_num)
    print('item_num', item_num)
    print("{}G data loaded.".format(get_memory_info()))
    datainfo = {}
    datainfo['user_num'] = user_num
    datainfo['item_num'] = item_num
    datainfo['user_review_length'] = review_len_u
    datainfo['item_review_length'] = review_len_i
    datainfo['user_review_word_num'] = len(user_vocab)
    datainfo['item_review_word_num'] = len(item_vocab)
    test_data = data_base+'music.test'
    train_data = data_base+'music.train'
    print("Start load train_data file ...")
    with open(train_data, 'rb') as f:
        train_data = pickle.load(f)
    print("done. {}G data loaded.".format(get_memory_info()))
    print("Start load valid_data file ...")
    with open(test_data, 'rb') as f:
        test_data = pickle.load(f)
    print("done. {}G data loaded.".format(get_memory_info()))
    train_dataset = Dataset(train_data, u_text, i_text)
    test_dataset  = Dataset(test_data, u_text, i_text)
    print("{}G data loaded.".format(get_memory_info()))
    return datainfo, train_dataset, test_dataset
    
def init_weights(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
        #module.weight.data.normal_(mean = 0.0, std = 0.05)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        #module.weight.data.normal_(mean = 0.0, std = 0.05)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.GRU):
        xavier_uniform_(module.weight_hh_l0)
        xavier_uniform_(module.weight_ih_l0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class TextEncoder(torch.nn.Module):
    def __init__(self, word_num, word_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.text_embedding = torch.nn.Embedding(word_num + 1,  word_dim)
        self.conv = torch.nn.Conv1d(word_dim, hidden_dim, kernel_size)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, text_seq):  #(b, num, len)
        text_batch = text_seq.shape[0]
        text_num   = text_seq.shape[1]
        text_seq = text_seq.reshape(text_batch * text_num, -1)  # (b*num, len)
        text_emb = self.text_embedding(text_seq) 
        text_emb = text_emb.permute(0, 2, 1)     #(b, hidden, l)
        text_conv = F.relu(self.conv(text_emb)).squeeze(-1)
        text_feat = self.max_pool(text_conv).squeeze(-1).reshape(text_batch, text_num, -1)
        return text_feat

class FeatAttention(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0):
        super().__init__()
        self.att_feat = torch.nn.Linear(hidden_dim, embedding_dim)
        self.att_target = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.att_layer  = torch.nn.Linear(embedding_dim, 1)
        self.top_linear = torch.nn.Linear(hidden_dim, embedding_dim)
        self.dropout    = torch.nn.Dropout(dropout)

    def forward(self, feat_emb, target_emb):
        feat_att = self.att_feat(feat_emb)
        target_att = self.att_target(target_emb)
        att_weight = self.att_layer(F.relu(feat_att + target_att))
        att_weight = F.softmax(att_weight, dim=1)
        att_out = (att_weight * feat_emb).sum(1)
        feature = self.dropout(att_out)
        feature = self.top_linear(feature)
        return feature
    
class Rating(torch.nn.Module):
    def __init__(self, embedding_dim, user_num, item_num):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)
        self.b_user = torch.nn.Embedding(user_num + 1, 1)
        self.b_item = torch.nn.Embedding(item_num + 1, 1)
    def forward(self, h, user_id, item_id):
        out = self.linear(h) + self.b_user(user_id.squeeze(-1)) + self.b_item(item_id.squeeze(-1))
        return out

class NARRE(torch.nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, hidden_dim, user_review_encoder, item_review_encoder):
        super().__init__()
        self.user_embeding = torch.nn.Embedding(user_num + 1, embedding_dim)
        self.item_embeding = torch.nn.Embedding(item_num + 1, embedding_dim)
        self.user_target_embeding = torch.nn.Embedding(item_num + 1, embedding_dim)
        self.item_target_embeding = torch.nn.Embedding(user_num + 1, embedding_dim)
        self.user_review_encoder = user_review_encoder
        self.item_review_encoder = item_review_encoder
        self.user_feat_attention = FeatAttention(embedding_dim, hidden_dim)
        self.item_feat_attention = FeatAttention(embedding_dim, hidden_dim)
        self.rating = Rating(embedding_dim, user_num, item_num)
        self.apply(init_weights)

       
    def forward(self, user_review, item_review, user_id, item_id, item_id_per_review, user_id_per_review):
        user_emb = self.user_embeding(user_id.squeeze(-1))
        item_emb = self.item_embeding(item_id.squeeze(-1))
        user_target = self.user_target_embeding(item_id_per_review) 
        item_target = self.item_target_embeding(user_id_per_review)
        user_feature = self.user_review_encoder(user_review) 
        item_feature = self.item_review_encoder(item_review)
        user_feature  = self.user_feat_attention(user_feature, user_target)
        item_feature  = self.item_feat_attention(item_feature, item_target)
        h = (user_emb + user_feature) * (item_emb + item_feature)
        out = self.rating(h, user_id, item_id)
        return out.squeeze(-1)

class MF(torch.nn.Module):
    def __init__(self, user_num, item_num, embedding_dim):
        super().__init__()
        self.user_embeding = torch.nn.Embedding(user_num + 1, embedding_dim)
        self.item_embeding = torch.nn.Embedding(item_num + 1, embedding_dim)
        self.rating = Rating(embedding_dim, user_num, item_num)
        self.apply(init_weights)
       
    def forward(self, user_review, item_review, user_id, item_id, item_id_per_review, user_id_per_review):
        user_emb = self.user_embeding(user_id.squeeze(-1))
        item_emb = self.item_embeding(item_id.squeeze(-1))
        h = user_emb * item_emb
        out = self.rating(h, user_id, item_id)
        return out.squeeze(-1)


class Trainer():
    def __init__(self, model, train_data, test_data, batch_size, logger):
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        print('device', self.device)
        self.model= model.to(self.device)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory = True)
        self.test_loader  = DataLoader(dataset=test_data,  batch_size=batch_size * 2, num_workers = 4, pin_memory = True)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.best_res = np.inf
        self.logger = logger
        self.es = EarlyStopper()
        self.loss = nn.MSELoss()
        self.save_file = "/code/r9/xai/NARRE/torch_version/{}.bin".format(self.model.name)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
    def run(self, epoch_num=10):
        for i in range(epoch_num):
            train_res = self.train(i)
            test_res  = self.test(i)
            self.logger.info("Epoch = {:d} ; {:s}; train mse loss = {:.4f} ; test mse loss = {:.4f}".format(i, self.model.name, train_res, test_res))
            if not self.es.is_continuable(test_res):
                print("Early stop at epoch {}".format(i))
                break

    def train(self, epoch):
        self.model.train()
        return self.run_step(epoch, self.train_loader, True)
    
    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()
        return self.run_step(epoch, self.test_loader, False)

    def run_step(self, epoch, data_loader, train=True):
        run_tag = "Train" if train else "Test"
        avg_loss  = 0.0
        avg_score = 0.0
        num_batch = len(data_loader)
        for i, batch in enumerate(data_loader):
            batch = [v.to(self.device) for v in batch]
            u_batch, i_batch, uid, iid, reuid, reiid, target = batch
            if train:
                score = self.model(u_batch, i_batch, uid, iid, reuid, reiid)
                loss  = self.loss(score, target.float().squeeze(-1))
                self.optim.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type=2)
                self.optim.step()            
            else:
                score = self.model(u_batch, i_batch, uid, iid, reuid, reiid)
                loss  = self.loss(score, target.squeeze(-1))
            avg_loss += loss.item()
            if i%100 == 0:
                print("Epoch {:d} \t {:s} \t avg_loss = {:.4f}".format(epoch, run_tag, avg_loss/num_batch))
        return np.sqrt(avg_loss / num_batch)

    def save(self, epoch):
        if hasattr(self.model, "module"):
            torch.save(self.model.module.state_dict(), self.save_file)
        else:
            torch.save(self.model.state_dict(), self.save_file)
        print("Model Saved on:{}".format(self.save_file))
        return file_path

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m",    type = str, default = "narre", help = "narre, mf")
parser.add_argument("--dataset", "-n",  type = str, default = "toys", help = "toys, kindle or music")
parser.add_argument("--out", "-o",  type = int, default = 1)
parser.add_argument("--epoch",   "-e",  type = int, default = 10)
parser.add_argument("--batch_size", "-b",  type = int, default=512)
parser.add_argument("--embedding_dim", "-dim", type = int, default = 128)
parser.add_argument("--num_epoch", "-epoch", type = int, default = 100)

args = parser.parse_args()
info_data, train_data, test_data = get_dataset(args.dataset)
print("Totally {}G data loaded.".format(get_memory_info()))
embedding_dim = args.embedding_dim
word_dim   = 256
hidden_dim = 64
user_review_encoder = TextEncoder(info_data['user_review_word_num'], word_dim, hidden_dim)
item_review_encoder = TextEncoder(info_data['item_review_word_num'], word_dim, hidden_dim)

model = get_model('narre', info_data['user_num'] + 3, info_data['item_num'] + 3, embedding_dim, hidden_dim, user_review_encoder, item_review_encoder)

logger = get_logger("./{}_{}.log".format(args.dataset, args.out))
trainer = Trainer(model, train_data, test_data, args.batch_size, logger)
trainer.run(args.epoch)

