import argparse
import copy
import csv
import os

import torch
import tqdm
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from glob import glob
from nets import nn
from utils import util
from CustomDataset import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
def csv_reader(enc_dec_mode=0):
    csv_features = [
        "내부 온도 1 평균",
        "내부 온도 1 최고",
        "내부 온도 1 최저",
        "내부 습도 1 평균",
        "내부 습도 1 최고",
        "내부 습도 1 최저",
        "내부 이슬점 평균",
        "내부 이슬점 최고",
        "내부 이슬점 최저",
    ]
    csv_files = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/train/*/*.csv"))

    # temp_csv = pd.read_csv(csv_files[0])[csv_features]
    # max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # # feature 별 최대값, 최솟값 계산
    # for csv in tqdm(csv_files[1:]):
    #     temp_csv = pd.read_csv(csv)[csv_features]
    #     temp_csv = temp_csv.replace("-", np.nan).dropna()
    #     if len(temp_csv) == 0:
    #         continue
    #     temp_csv = temp_csv.astype(float)
    #     temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
    #     max_arr = np.max([max_arr, temp_max], axis=0)
    #     min_arr = np.min([min_arr, temp_min], axis=0)

    # feature 별 최대값, 최솟값 dictionary 생성
    min_arr = np.array([3.4, 3.4, 3.3, 23.7, 25.9, 0, 0.1, 0.2, 0.0])
    max_arr = np.array([46.8, 47.1, 46.6, 100, 100, 100, 34.5, 34.7, 34.4])

    csv_feature_dict = {
        csv_features[i]: [min_arr[i], max_arr[i]] for i in range(len(csv_features))
    }
    # 변수 설명 csv 파일 참조
    crop = {"1": "딸기", "2": "토마토", "3": "파프리카", "4": "오이", "5": "고추", "6": "시설포도"}
    disease = {
        "1": {
            "a1": "딸기잿빛곰팡이병",
            "a2": "딸기흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "2": {
            "a5": "토마토흰가루병",
            "a6": "토마토잿빛곰팡이병",
            "b2": "열과",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "3": {
            "a9": "파프리카흰가루병",
            "a10": "파프리카잘록병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "4": {
            "a3": "오이노균병",
            "a4": "오이흰가루병",
            "b1": "냉해피해",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "5": {
            "a7": "고추탄저병",
            "a8": "고추흰가루병",
            "b3": "칼슘결핍",
            "b6": "다량원소결핍 (N)",
            "b7": "다량원소결핍 (P)",
            "b8": "다량원소결핍 (K)",
        },
        "6": {"a11": "시설포도탄저병", "a12": "시설포도노균병", "b4": "일소피해", "b5": "축과병"},
    }
    risk = {"1": "초기", "2": "중기", "3": "말기"}

    label_description = {}  # classification 111 number ex) '딸기_다량원소결핍 (P)_말기'

    label_description_crop = {}
    label_description_disease = {}
    label_description_risk = {}
    for key, value in disease.items():
        label_description[f"{key}_00_0"] = f"{crop[key]}_정상"
        for disease_code in value:
            for risk_code in risk:
                label = f"{key}_{disease_code}_{risk_code}"
                label_crop = f"{key}"
                label_disease = f"{disease_code}"
                label_risk = f"{risk_code}"

                label_description[
                    label
                ] = f"{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}"
                label_description_crop[label_crop] = f"{crop[key]}"
                label_description_disease[
                    label_disease
                ] = f"{disease[key][disease_code]}"
                label_description_risk[label_risk] = f"{risk[risk_code]}"

    label_description_disease["00"] = "정상"
    label_description_risk["0"] = "정상"

    # ex) '1_00_0' : 0
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_encoder_crop = {key: idx for idx, key in enumerate(label_description_crop)}
    label_encoder_disease = {
        key: idx for idx, key in enumerate(label_description_disease)
    }
    label_encoder_risk = {key: idx for idx, key in enumerate(label_description_risk)}

    # ex) '0' : '1_00_0'
    label_decoder = {val: key for key, val in label_encoder.items()}
    label_decoder_crop = {val: key for key, val in label_encoder_crop.items()}
    label_decoder_disease = {val: key for key, val in label_encoder_disease.items()}
    label_decoder_risk = {val: key for key, val in label_encoder_risk.items()}

    # print(label_decoder)
    if enc_dec_mode == 0:
        return csv_feature_dict, label_encoder, label_decoder
    else:
        return (
            csv_feature_dict,
            [label_encoder_crop, label_encoder_disease, label_encoder_risk],
            [label_decoder_crop, label_decoder_disease, label_decoder_risk],
        )


def batch(images, target, model, criterion=None):
    images = images.cuda()
    target = target.cuda()
    if criterion:
        with torch.cuda.amp.autocast():
            loss = criterion(model(images), target)
        return loss
    else:
        return util.accuracy(model(images), target, top_k=(1, 5))


def train(args,train_set,val_set):
    epochs = 200
    batch_size = 256
    util.set_seeds(args.rank)
    model = nn.EfficientNet(args).cuda()
    # lr = batch_size * torch.cuda.device_count() * 0.256 / 4096
    lr = 0.00001
    optimizer = nn.RMSprop(util.add_weight_decay(model), lr, 0.9, 1e-3, momentum=0.9)
    ema = nn.EMA(model)
    model.cuda(args.local_rank)

    _, label_encoder,label_decoder= csv_reader(enc_dec_mode=0)
    train_dataset = CustomDataset(train_set, label_encoder)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=16, shuffle=False)
    val_dataset = CustomDataset(val_set,label_encoder)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False)

    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = nn.StepLR(optimizer)
    amp_scale = torch.cuda.amp.GradScaler()

    if args.tf:
        last_name = 'last_tf'
        best_name = 'best_tf'
        step_name = 'step_tf'
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        last_name = 'last_pt'
        best_name = 'best_pt'
        step_name = 'step_pt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
    #                                transforms.Compose([util.RandomResize(),
    #                                                    transforms.ColorJitter(0.4, 0.4, 0.4),
    #                                                    transforms.RandomHorizontalFlip(),
    #                                                    util.RandomAugment(),
    #                                                    transforms.ToTensor(), normalize]))
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = None
    loader = train_loader
    with open(f'weights/{step_name}.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@1', 'acc@5'])
            writer.writeheader()
        best_acc1 = 0
        for epoch in range(0, epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                # bar = tqdm.tqdm(loader, total=len(loader))
                bar =tqdm(enumerate(loader))
            else:
                bar = loader
            model.train()
            for i, batch_item in bar:
                images=batch_item['img']
                target=batch_item['label']
                loss = batch(images, target, model, criterion)
                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                ema.update(model)
                torch.cuda.synchronize()
                if args.local_rank == 0:
                    bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))

            scheduler.step(epoch + 1)
            if args.local_rank == 0:
                acc1, acc5 = test(args, ema.model.eval(),val_dataloader)
                writer.writerow({'acc@1': str(f'{acc1:.3f}'),
                                 'acc@5': str(f'{acc5:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                state = {'model': copy.deepcopy(ema.model).half()}
                torch.save(state, f'weights/{last_name}.pt')
                if acc1 > best_acc1:
                    torch.save(state, f'weights/{best_name}.pt')
                del state
                best_acc1 = max(acc1, best_acc1)
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(args, model,val_dataloader):
    if model is None:
        if args.tf:
            model = torch.load('weights/best_tf.pt', map_location='cuda')['model'].float().eval()
        else:
            model = torch.load('weights/best_pt.pt', map_location='cuda')['model'].float().eval()

    if args.tf:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


   
    loader = val_dataloader
    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    with torch.no_grad():
        for i, batch_item in tqdm(enumerate(loader)):
            images=batch_item['img']
            target=batch_item['label']
            acc1, acc5 = batch(images, target, model)
            torch.cuda.synchronize()
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return acc1, acc5


def print_parameters(args):
    model = nn.EfficientNet(args).eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {int(params)}')


def benchmark(args):
    shape = (1, 3, 384, 384)
    util.torch2onnx(nn.EfficientNet(args).export().eval(), shape)
    util.onnx2caffe()
    util.print_benchmark(shape)


def main():
    # python -m torch.distributed.launch --nproc_per_node=3 main.py --train
    train2 = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/train/*"))
    test_set = sorted(glob("/DL_data_big/DACON_2022_LeafChallenge/leaf_data/data/test/*"))
    labelsss = pd.read_csv("/works/EfficientNet-PyTorch/examples/imagenet/train.csv")["label"]
    train_set, val_set = train_test_split(train2, test_size=0.2, stratify=labelsss)
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train', action='store_true',default=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tf', action='store_true')
    args = parser.parse_args()
    args.nproc_per_node=8
    args.distributed = False
    args.rank = 0
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    if args.local_rank == 0:
        print_parameters(args)
    if args.benchmark:
        benchmark(args)
    if args.train:
        train(args,train_set, val_set)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
