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

    writer = SummaryWriter("log")
    torch.cuda.set_device(args.device_id)

    model = CrossModal(vocab_size=args.vocab_size, 
            pretrain_path=args.pretrain_path).cuda()
    #model = torch.nn.DataParallel(model).cuda()
    criterion = RankLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
            lr=args.learning_rate)

    step = 0
    for epoch in range(args.epochs):
        train_reader = DataReader(args.vocab_path, args.train_data_path, args.image_path, 
                args.vocab_size, args.batch_size, is_shuffle=True)
        print("train reader load succ......")
        for train_batch in train_reader.batch_generator():
            query = torch.from_numpy(train_batch[0]).cuda()
            pos = torch.stack(train_batch[1], 0).cuda()
            neg = torch.stack(train_batch[2], 0).cuda()

            optimizer.zero_grad()
        
            left, right = model(query, pos, neg)
            loss = criterion(left, right).cuda()

            loss.backward()
            optimizer.step()
            if step == 0:
                writer.add_graph(model, (query, pos, neg))

            if step % 100 == 0:
                writer.add_scalar('Train/Loss', loss.item(), step)    

            if step % args.eval_interval == 0:
                print('Epoch [{}/{}], Step [{}] Loss: {:.4f}'.format(epoch + 1, 
                            args.epochs, step, loss.item()), flush=True)

            if step % args.save_interval == 0:
                # Save the model checkpoint
                torch.save(model.state_dict(), '%s/model.ckpt' % args.model_path)
            step += 1

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
    
