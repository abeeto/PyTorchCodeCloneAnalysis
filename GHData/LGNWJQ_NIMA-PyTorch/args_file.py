"""
file - args_file.py
用于设定所有参数
"""

import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_writer', type=str, default="WJQ", help="Name of code writer")
    # 数据相关参数
    parser.add_argument('--train_csv_file', type=str, default='./csv_file/train_labels.csv',
                        help='训练数据集的csv标签文件路径')
    parser.add_argument('--val_csv_file', type=str, default='./csv_file/val_labels.csv',
                        help='验证数据集的csv标签文件路径')
    parser.add_argument('--test_csv_file', type=str, default='./csv_file/test_labels.csv',
                        help='测试数据集的csv标签文件路径')
    parser.add_argument('--image_path', type=str, default='D:/MyDataset/AVA_dataset/images/', help='数据集图像路径')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader使用的cpu线程数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')

    # 训练相关参数
    parser.add_argument('--seed', type=int, default=2022, help='随机种子')
    parser.add_argument('--network', type=str, default='convnext',
                        help='神经网络类型，可选：vgg16、densenet121、convnext')
    parser.add_argument('--epochs', type=int, default=24, help='训练周期数')
    parser.add_argument('--stop_count', type=int, default=5, help='训练终止计数')

    parser.add_argument('--conv_base_lr', type=float, default=5e-3)
    parser.add_argument('--linear_lr', type=float, default=5e-4)

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')
    parser.add_argument('--warm_start_path', type=str,
                        default=None,
                        help='继续训练的路径')

    # 显示参数
    args = parser.parse_args()
    print('=-' * 30)
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('=-' * 30)

    return args


if __name__ == '__main__':
    args = set_args()
