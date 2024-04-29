from transformers import BertTokenizer, BertConfig, BertModel
from transformers import get_constant_schedule_with_warmup,AdamW,get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import math
from DataProcess import Mydataset,testset
from Evaluate import evaluate_valid,evaluate_test
from ModelModule import BertClassifical
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
#计算时间函数
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train_model(model,train_iter,valid_iter,args):

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    """
    optimizer = Adam([
    {'params': model.bert.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 3e-4}])
    """
    #监控学习率代码
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=1, min_lr=0.0001)
    #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=len(train_iter) // gradient_accumulation_steps * num_epochs)
    scheduler = get_constant_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=args.warmup_steps,)
    total_steps = 0
    best_f1 = 0
    stop_count = 0
    start_time = time.time()
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0

        for step,input in enumerate(train_iter):
            inputs = {
                "input_ids":input[0].cuda(non_blocking=True),
                "token_type_ids":input[1].cuda(non_blocking=True),
                "attention_mask":input[2].cuda(non_blocking=True)
            }
            total_steps+=1
            labels = input[3].cuda(non_blocking=True)
            logits = model(**inputs)
            #定义损失
            loss = F.cross_entropy(logits,labels)
            loss.backward()
            epoch_loss+=loss.item()
            #梯度累加
            if total_steps % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if total_steps % args.eval_steps ==0:
                end_time = timeSince(start_time)
                print("epoch: {},eval_steps: {},time: {}".format(epoch+1, total_steps, end_time))
                p,r,f1 = evaluate_valid(model, valid_iter)

                print("valid_p：{:.4f},valid_r：{:.4f},valid_f1:{:.4f}".format(p, r, f1))

                if f1 > best_f1:
                    best_f1 = f1
                    # 保存整个模型
                    #torch.save(model, 'resnet.pth')
                    #保存权重
                    torch.save(model.state_dict(), args.save_path)
                    # 释放显存
                    torch.cuda.empty_cache()
                    #model.train()
        #打印epoch_loss
        print('Epoch {} - Loss {:.4f}'.format(epoch + 1, epoch_loss / len(train_iter)))

if __name__=="__main__":


    import sys
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",default=-1,type=int,help="node rank for distributed training")
    parser.add_argument('--do_train', type=bool, default=None, help="")
    parser.add_argument('--do_test', type=bool, default=None, help="")
    parser.add_argument('--do_valid', type=bool, default=None, help="")
    parser.add_argument('--batch_size_per_gpu', type=int, default=4,help="")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="")
    parser.add_argument('--maxlen', type=int, default=265,help="")
    parser.add_argument('--warmup_steps', type=int, default=0,help="")
    parser.add_argument('--eval_steps', type=int, default=800, help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="")
    parser.add_argument('--save_path', type=str, default="/home/wq/model.pth", help="")
    parser.add_argument('--num_epochs', type=int, default=5, help="")
    parser.add_argument('--pretrained_model_path', type=str, default="/home/wq/ner/chinese_roberta_wwm_large_ext_pytorch/", help="")
    parser.add_argument('--seed_value', type=int, default=2020, help="")
    args = parser.parse_args()
    # 设置种子,保证能复现
    np.random.seed(args.seed_value)
    torch.manual_seed(args.seed_value)
    torch.cuda.manual_seed_all(args.seed_value)
    torch.backends.cudnn.deterministic = True
    print("打印args参数")
    print(args)

    # 读取数据集
    train = pd.read_csv("/home/wq/fcqa/train_data_kold5.csv")
    valid = pd.read_csv("/home/wq/fcqa/valid_data_kold5.csv")
    test = pd.read_csv("/home/wq/fcqa/test_data.csv")
    train = train[["question", "reply_data", "label"]]
    valid = valid[["question", "reply_data", "label"]]
    test_data = test[["question", "reply_data"]]
    # 加载预训练模型
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    #pre_model = BertModel.from_pretrained(pretrained_model_path)
    config = BertConfig.from_pretrained(args.pretrained_model_path)

    if torch.distributed.is_nccl_available():
        print("nccl is available")
        # 初始化使用nccl后端
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    else:
        print("nccl is unavailabel")
    #训练
    if args.do_train:
        #
        # 构建数据集
        train_set = Mydataset(train, tokenizer, args.maxlen)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_iter = DataLoader(dataset=train_set,
                                batch_size=args.batch_size_per_gpu,
                                shuffle=False,
                                pin_memory=True,
                                sampler=train_sampler)

        valid_set = Mydataset(valid, tokenizer, args.maxlen)
        valid_iter = DataLoader(dataset=valid_set, batch_size=args.batch_size_per_gpu, shuffle=False)

        model = BertClassifical(config,BertModel,args.pretrained_model_path)
        model.cuda()

        if torch.cuda.device_count() > 1:
            #不要用这种数据分布式,负载不均衡且训练慢的一批
            #model = nn.DataParallel(model,device_ids = [0,1])
            #换另一种分布式
            #model = nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],find_unused_parameters=True)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        #model.to(device)
        train_model(model, train_iter, valid_iter, args)
    if args.do_valid:
        start_time = time.time()
        valid_set = Mydataset(valid, tokenizer, args.maxlen)
        valid_iter = DataLoader(dataset=valid_set, batch_size=args.batch_size_per_gpu, shuffle=False)
        model = BertClassifical(config, BertModel, args.pretrained_model_path)
        model.cuda()
        #分布式训练保存的模型需要加上下面代码才能预测
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model.load_state_dict(torch.load(args.save_path))

        print("模型加载完成的时间为{}".format(timeSince(start_time)))
        p, r, f1 = evaluate_valid(model, valid_iter)
        print("验证集的模型精确率：{},召回率：{},f1值：{}".format(p,r,f1))
    if args.do_test:
        #加载整个模型
        #model = torch.load(save_path)
        #加载模型权重
        start_time = time.time()
        #print("开始预测的时间为{}".format(start_time))
        test_set = testset(test_data, tokenizer, args.maxlen)
        test_iter = DataLoader(dataset=test_set, batch_size=args.batch_size_per_gpu, shuffle=False)
        model = BertClassifical(config, BertModel, args.pretrained_model_path)
        model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model.load_state_dict(torch.load(args.save_path),False)

        print("模型加载完成的时间为{}".format(timeSince(start_time)))
        y_preds = evaluate_test(model,test_iter)
        print("预测完成的时间为{}".format(timeSince(start_time)))
        print("预测集的长度为:{}".format(len(y_preds)))
        test["label"] = y_preds
        # 生成提交文件
        test[["question_id", "reply_id", "label"]].to_csv("./torch_submission_kfold10.tsv", header=None, index=None, sep="\t")
    #执行代码
    #python - m torch.distributed.launch --nproc_per_node = 2 run_classifical.py --do_train=True
    #python - m torch.distributed.launch --nproc_per_node = 2 run_classifical.py --do_test=True









