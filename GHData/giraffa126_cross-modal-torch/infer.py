# coding: utf8
import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
import argparse
from model import CrossModal 
from model import RankLoss
from reader import DataReader
from tensorboardX import SummaryWriter

torch.manual_seed(1)

def train(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    torch.cuda.set_device(args.device_id)
    model = CrossModal(vocab_size=args.vocab_size, 
            pretrain_path=args.pretrain_path).cuda()
    model.load_state_dict(torch.load(args.model_path + "/model.ckpt"))
    model.eval()

    train_reader = DataReader(args.vocab_path, "./data/query.txt", args.image_path, 
            args.vocab_size, args.batch_size, is_shuffle=False)
    for train_batch in train_reader.extract_emb_generator():
        query = torch.from_numpy(train_batch).cuda()
        vec_list  = model.query_emb(query)
        for vec in vec_list:
            print(" ".join([str(round(x, 4)) for x in vec.cpu().detach().numpy()]))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--vocab_size", type=int, default=250000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--pretrain_path", type=str, default="./pretrain_model/model.ckpt")
    parser.add_argument("--image_path", type=str, default="./data/images")
    parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
    parser.add_argument("--test_data_path", type=str, default="./data/test.txt")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.pkl")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--save_interval", type=int, default="200")
    parser.add_argument("--eval_interval", type=int, default="5")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    train(args)
    
