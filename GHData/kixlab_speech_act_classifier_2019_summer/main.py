from src.dataset.vrm_dataset import VRMDataset, label_mapper_VRM
from src.dataset.swda_dataset import SWDADataset, label_mapper_SWDA, label_weight_SWDA
from src.utils.word2vec import word2vec
from src.model.bilstm_crf import build_bilstm_crf
from src.model.bilstm_ram import build_bilstm_ram, ChunkCrossEntropyLoss, ChunkCrossEntropyLoss_VRM
from src.config import bilstm_ram_config,bilstm_ram_vrm_config,bilstm_crf_config
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import nltk
import os
import numpy as np
import pandas as pd
import argparse
import csv
import tqdm
import pdb

PRECISION_THRESHOLD = 0.70
ignore_index = -1

def str2bool(s):
    return s.lower() in ('true', '1')

parser = argparse.ArgumentParser(
    description='VRM Recognition Training With Pytorch')

parser.add_argument('--command', default="train",
                    help="train or test")
parser.add_argument('--net', default="BiLSTM-RAM",
                    help="The network architecture")
parser.add_argument('--multiplier', default=1, type=int,
                    help='hidden size multiplier')


# Params for Optimizer
parser.add_argument('--optim',default='SGD',
                    help='Optimizer type')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay for optim')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for optim')
parser.add_argument('--beta', default=1e-5, type=float,
                    help='Beta for optim')
parser.add_argument('--betas_0', default=0.9, type=float,
                    help='Betas for Adam')
parser.add_argument('--betas_1', default=0.999, type=float,
                    help='Betas for Adam')


# Params for loading pretrained basenet or checkpoints.


# Params for loading dataset
parser.add_argument('--dataset', default='SWDA',
                    help='dataset configuration')
parser.add_argument('--data_path',
                    help='dataset path')
parser.add_argument('--val_data_path',
                    help='dataset path')
parser.add_argument('--embedding_path',help='get embedding from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="40,60", type=str,
                    help="milestones for MultiStepLR. percentage values.")
parser.add_argument('--fixed_milestones')
# Params for Cosine Annealing
parser.add_argument('--t_max', default=100, type=float,
                    help='T_max value for Cosine Annealing Scheduler. percentage values')

# Train and Test params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=1.0, type=float,
                    help='the number epochs')
parser.add_argument('--logdir', default='log/',
                    help='Directory for logging')
parser.add_argument('--log_stride', default=100, type=int,
                    help='Logging steps')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--checkpoint_folder', default='checkpoint/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--checkpoint_stride', default=10, type=int,
                     help='saving model epochs')
parser.add_argument('--output_folder', default='data/',
                        help='Directory for saving test output')
parser.add_argument('--checkpoint_path', default=None,
                    help='Explicit Checkpoint path input by user')

parser.add_argument('--resume_point', default=None)


parser.add_argument('--testby_conv', default=None)
parser.add_argument('--use_conv_val', default=None)


args = parser.parse_args()


# not supporting multi-gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

def train(model,
            device,
            dataloader,
            criterion,
            optimizer,
            scheduler,
            writer,
            checkpoint_dir,
            epoch_size,
            log_stride,
            checkpoint_stride,
            val_epoch_size=10,
            val_dataloader=None,
            use_chunk=False,
            is_vrm=False,
            is_crf=False):

    best_precision = 0
    best_model_state = None
    previous_lr = None
    for epoch in tqdm.tqdm(range(epoch_size)):

        model.train()
        scheduler.step()
        average_loss = 0
        last_log = 0

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar("lr/train",current_lr,epoch)

        if previous_lr and previous_lr != current_lr:
            model.load_state_dict(best_model_state)

        previous_lr = current_lr

        total = len(dataloader)

        if val_epoch_size < 1 :
            val_stride = int(total*val_epoch_size)

        for n, data in enumerate(tqdm.tqdm(dataloader)):
            optimizer.zero_grad()

            utterances, labels = model.formatter(data)

            utterances = [utterance.to(device) for utterance in utterances]
            labels = labels.to(device)
            preds = model(utterances)

            loss = criterion(preds, labels)
            average_loss += loss

            loss.backward()
            optimizer.step()

            if (n+1) % log_stride == 0:

                last_log = (average_loss/log_stride).cpu().item()
                writer.add_scalar("loss/train", last_log, epoch*len(dataloader)+n)
                average_loss = 0

            if val_epoch_size < 1:

                if (n+1) % val_stride == 0:
                    precision = val_in_train(model,
                            device,
                            val_dataloader,
                            writer,
                            epoch*total+n,
                            use_chunk=use_chunk,
                            is_vrm=is_vrm,
                            is_crf=is_crf)

                    if precision > PRECISION_THRESHOLD and precision > best_precision:
                        best_precision = precision
                        best_model_state = model.state_dict()
                        torch.save(
                            model.state_dict(),
                            os.path.join(checkpoint_dir, "checkpoint_{}_{}.pth".format(precision,epoch)),
                        )

                    model.train()

        if val_epoch_size > 1 and val_dataloader:

            if (epoch+1)% val_epoch_size == 0:
                precision = val_in_train(model,
                            device,
                            val_dataloader,
                            writer,
                            (epoch+1)*total,
                            use_chunk=use_chunk,
                            is_vrm=is_vrm,
                            is_crf=is_crf)

                if precision > PRECISION_THRESHOLD and precision > best_precision:
                    best_precision = precision
                    best_model_state = model.state_dict()
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_dir, "checkpoint_{}_{}.pth".format(precision,epoch)),
                    )


        if (epoch+1)% checkpoint_stride == 0:
            torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(epoch)),
                )


    torch.save(model.state_dict(), os.path.join(checkpoint_dir,"checkpoint_{}.pth".format(epoch_size-1)))
    writer.close()



def val(model,
        device,
        dataloader,
        output_path,
        is_crf=False):

    # implemented only for SWDA.

    f = open(output_path, mode='w')
    wr = csv.writer(f)
    wr.writerow(['conversation_id','utterance_id','gt','pred'])

    model.eval()

    val_cid = []
    val_uid = []
    val_targ = []
    val_predict = []

    for n, data in enumerate(tqdm.tqdm(dataloader)):


        utterances, labels = model.formatter(data)
        utterances = [utterance.to(device) for utterance in utterances]
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(utterances)


        for i, pred in enumerate(preds):

            label = labels[0][i].item()

            if label == ignore_index:
                continue
            else:
                if is_crf:
                    predicted_label = pred
                else:
                    _, predicted_label = pred.max(-1)
                    predicted_label = predicted_label.item()

            val_cid.append(n)
            val_uid.append(i)
            val_targ.append(predicted_label)
            val_predict.append(label)


    for cid,uid,gt,pred in zip(val_cid,val_uid,val_targ,val_predict):

        wr.writerow([cid,uid,gt,pred])

    f.close()

def val_in_train(model,
        device,
        dataloader,
        writer,
        iter_num,
        use_chunk=False,
        is_vrm=False,
        is_crf=False):

    if is_vrm:
        return val_in_train_vrm(model,
                                device,
                                dataloader,
                                writer,
                                iter_num
                                )
    model.eval()

    correct_num = 0
    total_num = 0

    for n, data in enumerate(tqdm.tqdm(dataloader)):

        utterances, labels = model.formatter(data)

        utterances = [utterance.to(device) for utterance in utterances]
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(utterances)

        for i, pred in enumerate(preds):

            label = labels[0][i].item()

            if label == ignore_index:
                continue
            else:
                if is_crf:
                    predicted_label = pred
                else:
                    _, predicted_label = pred.max(-1)
                    predicted_label = predicted_label.item()
            if predicted_label == label:
                correct_num +=1
            total_num +=1


    _val_precision = correct_num / (total_num+0.00001)
    writer.add_scalar("val_precision/train", _val_precision, iter_num)

    return _val_precision

def val_in_train_vrm(model,
        device,
        dataloader,
        writer,
        iter_num
        ):

    model.eval()

    correct_nums = [0,0,0,0,0,0]
    total_nums = [0,0,0,0,0,0]

    for n, data in enumerate(tqdm.tqdm(dataloader)):


        utterances, labels = model.formatter(data)

        utterances = [utterance.to(device) for utterance in utterances]
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(utterances)

        for axis in range(6):
            _labels = labels[...,axis]
            _preds = [p[...,2*axis:2*axis+2] for p in preds]

            for i, pred in enumerate(_preds):
                label = _labels[0][i].item()
                if label ==ignore_index:
                    continue
                else:
                    _, predicted_label = pred.max(-1)
                    predicted_label = predicted_label.item()
                if predicted_label == label:
                    correct_nums[axis] += 1
                total_nums[axis] +=1

    total_prec= 0

    val_list = ['form_SE','form_PE','form_FR','intent_SE','intent_PE','intent_FR']
    for axis in range(6):
        _val_precision = correct_nums[axis] / (total_nums[axis]+0.00001)
        if writer:
            writer.add_scalar("val_precision/train"+str(axis), _val_precision, iter_num)
        else :
            print('precision'+val_list[axis]+str(_val_precision))
        total_prec += _val_precision

    total_prec = total_prec / 6.0
    if writer:
        writer.add_scalar("val_precision/train_total", total_prec, iter_num)
    return total_prec

def get_latest_version(checkpoint_dir,work_id=None):

    ls = os.listdir(checkpoint_dir)

    if work_id:
        versions = [int(chk.split('_')[-1].split('.')[0]) for chk in ls if chk.startswith(work_id)]
    else:
        versions= [int(chk.split('_')[-1].split('.')[0]) for chk in ls]

    version = 0
    if len(versions) != 0:
        versions.sort()
        version = versions[-1]
    return version


if __name__=='__main__':

    work_id = args.net +"_"+ args.dataset
    print(args.command + " on " + work_id)

    val_dataset = None
    by_conversation = False
    valby_conv = False
    lastpooling = False
    is_vrm = False
    is_crf = False

    if args.use_conv_val:
        valby_conv = True


    if args.net == 'BiLSTM-RAM':
        config = bilstm_ram_config
        if args.dataset == 'VRM':
            config = bilstm_ram_vrm_config
        builder = build_bilstm_ram
        by_conversation = config.by_conversation
        criterion = ChunkCrossEntropyLoss(num_chunk= config.chunk_size, weights=config.weights,ignore_index= ignore_index)

    elif args.net == 'BiLSTM-CRF':
        # this don't support VRM version
        config = bilstm_crf_config
        builder = build_bilstm_crf
        lastpooling = True
        by_conversation = True
        is_crf = True
        criterion = model.loss
    else :
        assert()
        config = None
        builder = None

    if args.testby_conv:
        by_conversation = True


    print("setting model...")

    model = builder(DEVICE,config,args.multiplier)
    model.to(DEVICE)

    print("getting embedding...")
    wv = word2vec(args.embedding_path,embedding_len=config.embedding_len, sent_len=config.sent_len)
    to_vector = wv.to_vector

    if args.dataset == 'SWDA':

        print("getting data...")
        df = pd.read_csv(args.data_path, error_bad_lines=False)
        print("setting dataset...")

        label_mapper = label_mapper_SWDA

        dataset= SWDADataset(df,to_vector,label_mapper,config.sent_len, config.pos_len,config.max_dialogue_len,
                    chunk_size=config.chunk_size,by_conversation=by_conversation,lastpooling=lastpooling)

        if args.val_data_path:
            val_df = pd.read_csv(args.val_data_path, error_bad_lines=False)
            val_dataset = SWDADataset(val_df,to_vector,label_mapper,config.sent_len,config.pos_len,config.max_dialogue_len,
                        chunk_size=config.chunk_size,by_conversation=valby_conv,lastpooling=lastpooling)

    elif args.dataset == 'VRM':

        assert(args.net != 'BiLSTM-CRF')

        is_vrm = True

        print("getting data...")
        df = pd.read_csv(args.data_path, error_bad_lines=False)
        print("setting dataset...")

        label_mapper = label_mapper_VRM

        dataset= VRMDataset(df,to_vector,label_mapper,config.sent_len, config.pos_len, config.max_dialogue_len,
                    chunk_size=config.chunk_size,by_conversation=by_conversation)

        if args.val_data_path:
            val_df = pd.read_csv(args.val_data_path, error_bad_lines=False)
            val_dataset = VRMDataset(val_df,to_vector,label_mapper,config.sent_len,config.pos_len,config.max_dialogue_len,
                    chunk_size=config.chunk_size,by_conversation=by_conversation)

        criterion = ChunkCrossEntropyLoss_VRM(num_chunk= config.chunk_size, weights=config.weights,ignore_index= ignore_index)

    else:
        assert()
        df = None
        dataset = None


    if args.command =='train':

        if args.resume_point:
            model.load_state_dict(torch.load(args.resume_point))

        dataloader = DataLoader(dataset,batch_size = args.batch_size,
                        num_workers=args.num_workers,shuffle=True)

        if val_dataset:
            val_dataloader = DataLoader(val_dataset,batch_size =1,
                        num_workers=1,shuffle=False)
        else:
            val_dataloader= None

        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.betas_0,args.betas_1),
                                weight_decay=args.weight_decay)
        elif args.optim == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else :
            assert()
            optimizer = None

        if args.scheduler == 'multi-step':
            if args.fixed_milestones:
                milestones = [ int(v.strip()) for v in args.milestones.split(",")]
            else:
                milestones = [ int((int(v.strip())/100)*args.num_epochs) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,gamma=args.gamma)

        elif args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, int(args.t_max/100*args.num_epochs))

        else :
            assert()
            scheduler = None

        chk_version = get_latest_version(args.checkpoint_folder,work_id)+1
        log_version = get_latest_version(args.logdir,work_id)+1

        checkpoint_dir = os.path.join(args.checkpoint_folder,work_id+"_"+str(chk_version))
        os.mkdir(checkpoint_dir)
        log_dir = os.path.join(args.logdir,work_id+"_"+str(log_version))

        writer = SummaryWriter(log_dir)

        print("start training...")
        train(model,
            DEVICE,
            dataloader,
            criterion,
            optimizer,
            scheduler,
            writer,
            checkpoint_dir=checkpoint_dir,
            epoch_size=args.num_epochs,
            log_stride=args.log_stride,
            checkpoint_stride=args.checkpoint_stride,
            val_epoch_size=args.validation_epochs,
            val_dataloader=val_dataloader,
            is_vrm=is_vrm,
            is_crf=is_crf)


    elif args.command =='test':


        dataloader = DataLoader(dataset,batch_size = 1,
                        num_workers=args.num_workers,shuffle=False)
        if args.checkpoint_path:
            chk_path = args.checkpoint_path
            output_path = os.path.join(args.output_folder,work_id+"_comm_ouptut.csv")

            model.load_state_dict(torch.load(chk_path))
        else:
            latest_work = get_latest_version(args.checkpoint_folder,work_id)
            while latest_work >= 0:

                work_dir = os.path.join(args.checkpoint_folder,work_id+"_"+str(latest_work))
                if len(os.listdir(work_dir))==0:
                    latest_work -=1
                    continue
                else:
                    break

            if latest_work >= 0:
                print("loading latest checkpoint...")
                latest_epoch = get_latest_version(work_dir)
                chk_path = os.path.join(work_dir,"checkpoint_{}.pth".format(latest_epoch))
                model.load_state_dict(torch.load(chk_path))
            else :
                latest_work= 999

            output_path = os.path.join(args.output_folder,work_id+"_"+str(latest_work)+"_ouptut.csv")

        print("running on test set...")

        if is_vrm:
            precision = val_in_train_vrm(model,
                                        DEVICE,
                                        dataloader,
                                        None,
                                        0)
            print( "average preicsion : " + str(precision))

        else:
            val(model,DEVICE,dataloader,output_path,is_crf=is_crf)

            print("Evaulation of "+work_id)
            df= pd.read_csv(output_path)
            val_targ = df["gt"]
            val_predict = df["pred"]

            average = 'micro'

            _val_f1 = f1_score(val_targ, val_predict,average=average)
            _val_recall = recall_score(val_targ, val_predict,average=average)
            _val_precision = precision_score(val_targ, val_predict,average=average)

            print("f1",_val_f1)
            print("recall",_val_recall)
            print("precision",_val_precision)


