import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from network import NIMA_Dict
from Dataset.dataset import AVADataset, val_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./weight/NIMA_vgg16.pt',
                        # default='./weight/NIMA_dense121.pt',
                        # default='./weight/NIMA_convnext.pt',
                        help='保存预训练模型的路径')
    parser.add_argument('--test_csv_file', type=str, default='./csv_file/test_labels.csv',
                        help='测试数据集的csv标签文件路径')
    parser.add_argument('--image_path', type=str, default='D:/MyDataset/AVA_dataset/images/',
                        help='数据集图像路径')
    parser.add_argument('--network', type=str, default='vgg16',
                            help='神经网络类型，可选：vgg16、densenet121、convnext')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_path', type=str, default='./test_result',
                        help='测试结果的保存路径')
    parser.add_argument('--save_name', type=str, default='vgg16111.txt',
                        help='测试结果的文件名')
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = NIMA_Dict[args.network].to(device)
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()

    test_set = AVADataset(
        csv_file_path=args.test_csv_file,
        image_path=args.image_path,
        transform=val_transform,
    )
    test_loader = DataLoader(
        dataset=test_set,
        num_workers=4,
        batch_size=1,
        shuffle=False
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, args.save_name), 'w') as f:
        f.write(args.network + '\n')

    count0_25 = 0.0
    count0_5 = 0.0
    count1 = 0.0
    count2 = 0.0
    count3 = 0.0
    score_list = torch.arange(1, 11, dtype=torch.float32).reshape(10, 1).to(device)

    loop = tqdm(test_loader)
    for i, sample in enumerate(loop):
        images = sample['image'].to(device)
        labels = sample['label'].to(device).float()
        name = sample['image_name']

        gt_mean = labels@score_list
        gt_d = labels@(score_list - gt_mean)**2
        gt_std = torch.sqrt(gt_d)
        with torch.no_grad():
            out_p = net(images)
        pre_mean = out_p@score_list
        pre_d = out_p@(score_list - pre_mean)**2
        pre_std = torch.sqrt(pre_d)

        loop.set_postfix(
            gt_mean=gt_mean.item(),
            pre_mean=pre_mean.item()
        )
        loop.set_description(args.network)

        if (pre_mean >= gt_mean-gt_std) and (pre_mean <= gt_mean+gt_std):
            count1 += 1
        if (pre_mean >= gt_mean-2*gt_std) and (pre_mean <= gt_mean+2*gt_std):
            count2 += 1
        if (pre_mean >= gt_mean-3*gt_std) and (pre_mean <= gt_mean+3*gt_std):
            count3 += 1
        if (pre_mean >= gt_mean-0.5*gt_std) and (pre_mean <= gt_mean+0.5*gt_std):
            count0_5 += 1
        if (pre_mean >= gt_mean-0.25*gt_std) and (pre_mean <= gt_mean+0.25*gt_std):
            count0_25 += 1

        write_str = name[0] + ' mean: %.3f | std: %.3f | GT: %.3f | %.3f\n' % \
                    (pre_mean.item(), pre_std.item(), gt_mean.item(), gt_std.item())
        with open(os.path.join(args.save_path, args.save_name), 'a') as f:
            f.write(write_str)

    with open(os.path.join(args.save_path, args.save_name), 'a') as f:
        f.write('result0_25: {}\n'.format(count0_25 / test_set.__len__()))
        f.write('result0_5: {}\n'.format(count0_5 / test_set.__len__()))
        f.write('result1: {}\n'.format(count1 / test_set.__len__()))
        f.write('result2: {}\n'.format(count2 / test_set.__len__()))
        f.write('result3: {}\n'.format(count3 / test_set.__len__()))

    print('result0_25: {}'.format(count0_25 / test_set.__len__()))
    print('result0_5: {}'.format(count0_5 / test_set.__len__()))
    print('result1: {}'.format(count1 / test_set.__len__()))
    print('result2: {}'.format(count2 / test_set.__len__()))
    print('result3: {}'.format(count3 / test_set.__len__()))
    print('over')




if __name__ == '__main__':
    main()

