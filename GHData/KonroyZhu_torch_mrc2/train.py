import argparse
import json
import pickle
import time

import torch

from com.utils import shuffle_data, padding, pad_answer, get_model_parameters
from models.mwan_full import MwAN_full
from models.phn import PHN
from models.qa_net import QA_Net
from preprocess.get_emb import get_emb_mat


parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling '
                                             'Sentence Pairs of the AI-Challenges')
parser.add_argument('--save', type=str, default='net/mwan_f.pt',
                    help='path to save the final model')
args = parser.parse_args()


def train(epoch, net,train_dt, opt, best):
    net.train()
    data = shuffle_data(train_dt, 1)
    total_loss = 0.0
    time_sum=0.0
    for num, i in enumerate(range(0, len(data), opts["batch"])):
        time_start = time.time()
        one = data[i:i + opts["batch"]]
        query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
        passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
        answer = pad_answer([x[2] for x in one])
        ids = [x[3] for x in one]
        query, passage, answer, ids = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer),ids
        if torch.cuda.is_available():
            query = query.cuda()
            passage = passage.cuda()
            answer = answer.cuda()
        opt.zero_grad()
        loss = net([query, passage, answer,ids, True,True])
        loss.backward()
        total_loss += loss.item()
        opt.step()
        # 计时
        time_end = time.time()
        cost = (time_end - time_start)
        time_sum += cost
        if (num + 1) % opts["log_interval"] == 0:
            ts=str('%.2f' % time_sum)
            print('|---epoch {:d} train error is {:f}  eclipse {:.2f}%  costing: {} best {} ---|'.format(epoch,
                                                                  total_loss /opts["log_interval"],
                                                                  i * 100.0 / len( data),ts + " s",best))
            time_sum=0.0
            total_loss = 0


def test(net, valid_data):
    net.eval()
    r, a = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(valid_data), opts["batch"]):
            print("{} in {}".format(i, len(valid_data)))
            one = valid_data[i:i + opts["batch"]]
            query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
            passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
            answer = pad_answer([x[2] for x in one],max_len=opts["alt_len"])
            ids = [x[3] for x in one]
            query, passage, answer,ids = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer),ids
            if torch.cuda.is_available():
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = net([query, passage, answer,ids, False , True])
            r += torch.eq(output, 0).sum().item()
            a += len(one)
    return r * 100.0 / a


def main():
    best = 0.0
    for epoch in range(opts["epoch"]):
        train(epoch,model,train_data,optimizer,best)
        acc = test(net=model,valid_data=dev_data)
        if acc > best:
            best = acc
            with open(args.save, 'wb') as f:
                torch.save(model, f)
        print ('epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best))


if __name__ == '__main__':
    opts = json.load(open("models/config.json"))
    #embedding_matrix=torch.FloatTensor(get_emb_mat("data/emb/id2v.pkl"))
    embedding_matrix = None
    opts["dep_path"]="data/dep/merged.fiedler.pkl"
    """ toggle for PHN and others
    opts["dropout"]=0.5 # for PHN 
    model=PHN(opts,embedding_matrix) # 13406161
    """
    opts["dropout"] = 0.2  # for QA & MwAN
    opts["head_size"]=1
    # """ """toggle for QA/MwAN """
    model=QA_Net(opts,embedding_matrix)  # 14844161
    """
    model = MwAN_full(opts, embedding_matrix)  # 16821760
    # """
    print('Model total parameters:', get_model_parameters(model))
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adamax(model.parameters())

    with open(opts["data"] + 'train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open(opts["data"] + 'dev.pickle', 'rb') as f:
        dev_data = pickle.load(f)
    dev_data = sorted(dev_data, key=lambda x: len(x[1]))

    print('train data size {:d}, dev data size {:d}'.format(len(train_data), len(dev_data)))

    main()


