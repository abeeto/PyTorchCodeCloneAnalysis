import argparse
import os
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torch.utils.data import DataLoader

from network import NIMA_Dict
from Dataset.dataset import AVADataset, val_transform


class ImageSet(Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        self.images_path = images_path
        self.transform = transform
        self.image_list = os.listdir(images_path)
        if image_num is not None:
            self.image_list = self.image_list[:image_num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_name = os.path.join(self.images_path, self.image_list[item])
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = {
            'image': image,
            'name': self.image_list[item]
        }
        return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./weight/NIMA_vgg16.pt',
                        # default='./weight/NIMA_dense121.pt',
                        # default='./weight/NIMA_convnext.pt',
                        help='保存预训练模型的路径')
    parser.add_argument('--image_path', type=str, default='D:/MyDataset/AVA_dataset/images/',
                        help='数据集图像路径')
    parser.add_argument('--network', type=str, default='vgg16',
                            help='神经网络类型，可选：vgg16、densenet121、convnext')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_path', type=str, default='./test_result111',
                        help='测试结果的保存路径')
    parser.add_argument('--save_name', type=str, default='vgg16.txt',
                        help='测试结果的文件名')
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = NIMA_Dict[args.network].to(device)
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_set = ImageSet(
        images_path=args.image_path,
        transform=test_transform,
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

    score_list = torch.arange(1, 11, dtype=torch.float32).reshape(10, 1).to(device)

    loop = tqdm(test_loader)
    for i, sample in enumerate(loop):
        image = sample['image'].to(device)
        name = sample['name']
        with torch.no_grad():
            out_p = net(image)
        pre_mean = out_p@score_list
        pre_d = out_p@(score_list - pre_mean)**2
        pre_std = torch.sqrt(pre_d)

        loop.set_postfix(
            pre_mean=pre_mean.item()
        )
        loop.set_description(args.network)

        write_str = name[0] + '| mean: %.3f | std: %.3f |' % \
                    (pre_mean.item(), pre_std.item())
        with open(os.path.join(args.save_path, args.save_name), 'a') as f:
            f.write(write_str)
            for i in range(10):
                f.write('%.4f ' % out_p[0, i].item())
            f.write('\n')




if __name__ == '__main__':
    main()

