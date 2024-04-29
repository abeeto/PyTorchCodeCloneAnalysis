import argparse
import time

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import (DALIClassificationIterator,
                                        DALIGenericIterator)
from torch import optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from tqdm import tqdm

import model

DOG_CAT_PATH = "data/dogcat"
CROP_SIZE = 224

# Training settings
parser = argparse.ArgumentParser(description = 'NVIDIA DALI Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

cudnn.benchmark = True


class HybridTrainPipe(Pipeline):
    """
    FileReader:
        to read files from the drive
    HostDecoder:
        to decode images to RGB format
    """
    def __init__(self,
        batch_size,
        num_threads,
        device_id,
        data_dir,
        crop = 224,
        dali_cpu = False
    ):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads,
                                              device_id, seed = 12 + device_id)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True, initial_fill = 23000)
        # let user decide which pipeline works him
        if dali_cpu:
            dali_device = 'cpu'
            self.decode = ops.HostDecoder(device = dali_device, output_type = types.RGB)
        else:
            dali_device = 'gpu'
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)

        self.rrc = ops.RandomResizedCrop(device = dali_device, size = (crop, crop))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        self.labels = self.labels.gpu()
        return [images, self.labels]


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()

    for batch_idx, data in tqdm(enumerate(train_loader)):
        target = data[0]['label'].squeeze().cuda(6, non_blocking = True).long()
        data = data[0]["data"].cuda(6, non_blocking=True).type(torch.cuda.FloatTensor)
        data = data.permute(0, 3, 1, 2)
        data_var = Variable(data)
        target_var = Variable(target)

        optimizer.zero_grad()
        output = model(data_var)
        loss = criterion(output, target_var)

        # Compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        if (batch_idx + 1) % args.log_interval == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), 23000,
               100. * batch_idx / 23000, loss.item()))

# For basic dataloader
def train_basic(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for _, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def main():
    global args, DOG_CAT_PATH, CROP_SIZE
    args.world_size = 1

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:6" if use_cuda else "cpu")
    net = model.resnet50(num_classes = 2).to(device)

    # Define loss function and optimizer
    criterion = nn.functional.cross_entropy
    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:6" if use_cuda else "cpu")

    # Define Training Arguments
    train_kwargs = {'batch_size' : args.batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)

    # DALI Loader
    pipe = HybridTrainPipe(batch_size = args.batch_size,
                           num_threads = 4,
                           device_id = 6,
                           data_dir = DOG_CAT_PATH
    )
    pipe.build()

    # DALI
    train_loader_dali = DALIClassificationIterator(pipe, size = 23000)

    # Basic
    transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor()
    ])
    dogcat_dataset = ImageFolder(DOG_CAT_PATH, transform)
    train_loader = DataLoader(dogcat_dataset, batch_size=args.batch_size)

    # Common
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = nn.functional.cross_entropy

    # Start epoch
    for epoch in range(1, args.epochs + 1):
        train(args, net, device, train_loader_dali, criterion, optimizer, epoch)
        train_basic(args, net, device, train_loader, criterion, optimizer, epoch)
        scheduler.step(1)
        break

    start = time.time()
    for i, (data,target) in tqdm(enumerate(train_loader)):
        data, target = data.to('cuda:6'), target.to('cuda:6')
    test_time = time.time() - start
    print(f"Pytorch dataloader cost time: {test_time}")

    start = time.time()
    for i, data in tqdm(enumerate(train_loader_dali)):
        target = data[0]['label'].squeeze().cuda(6, non_blocking = True).long()
        data = data[0]["data"]
        data = data.permute(0, 3, 1, 2)
        data = data.type(torch.cuda.FloatTensor)
    test_time = time.time() - start
    print(f"Dali cost time: {test_time}")
    if args.save_model:
        torch.save(net.state_dict(), "DALI_final.pt")


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    main()