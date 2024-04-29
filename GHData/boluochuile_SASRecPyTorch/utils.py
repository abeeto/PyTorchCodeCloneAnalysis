import math
import numpy as np
from collections import defaultdict
import faiss
import torch


def evaluate_full(test_data, model, args, item_cate_map, save=True, coef=None):

    topN = args.topN
    item_embs = model.output_item().weight.data.cpu().numpy()

    try:
        cpu_index = faiss.IndexFlatL2(args.hidden_units)
        cpu_index.add(item_embs)
    except Exception as e:
        print(e)
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for src, tgt in test_data:

        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

        user_embs = model.predict(np.array(hist_item)).data.cpu().numpy()

        # 单兴趣表示[batch, embedding_dim]
        if len(user_embs.shape) == 2:
            # I: itemList    D:distance
            D, I = cpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list = set(I[i])
                for no, iid in enumerate(iid_list):
                    if iid in item_list:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            # I, D = item_embs_index.get_nns_by_vector(user_embs, topN, include_distances=True)
            D, I = cpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                if coef is None:
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        max_index = 0
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:
                                break
                        item_list_set.add(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                for no, iid in enumerate(iid_list):
                    if iid in item_list_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1

        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total

    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            # 先移除字符串首尾的空格再按‘,’切分
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def sample_softmax_loss(item_eb, pos_id, item_count, user_eb, hist_item):

    loss = torch.nn.CrossEntropyLoss()

    # (b, 11)
    sampled_logits = None
    pos_indices = torch.LongTensor(pos_id).to('cuda')
    pos_emb = item_eb(pos_indices)

    for i in range(user_eb.shape[0]):

        sampled_item = [random_neq(1, item_count, hist_item) for _ in range(10)]
        sampled_item_emb = item_eb(torch.from_numpy(np.array(sampled_item)).long().cuda())
        interest = user_eb[i]
        if sampled_logits is None:
            # (11, dim) * (dim, ) = (11, )
            sampled_logits = torch.matmul(torch.cat((pos_emb[i].view(1, -1), sampled_item_emb), 0), interest).view(1, -1)
        else:
            temp = torch.matmul(torch.cat((pos_emb[i].view(1, -1), sampled_item_emb), 0), interest).view(1, -1)
            sampled_logits = torch.cat((sampled_logits, temp), 0)

    labels = torch.zeros(user_eb.shape[0]).cuda()
    return loss(sampled_logits, labels.long())

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t