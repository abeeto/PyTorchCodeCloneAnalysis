"""Train Inv_Conv_Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import Glow
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/CIFAR-10')                                                ###############
# torch.cuda.empty_cache()
def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))
    torch.cuda.empty_cache()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # No normalization applied, since Glow expects inputs in (0, 1)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)



    # # transforms
    # transform_visualize = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))])

    # # datasets
    # trainset_visualize = torchvision.datasets.CIFAR10('./data',
    #     download=True,
    #     train=True,
    #     transform=transform_visualize)
    # def select_n_random(data, labels, n=100):
    #     '''
    #     Selects n random datapoints and their corresponding labels from a dataset
    #     '''
    #     assert len(data) == len(labels)

    #     perm = torch.randperm(len(data))
    #     return data[perm][:n], labels[perm][:n]

    # # select random images and their target indices
    # images, labels = select_n_random(trainset_visualize.data, trainset_visualize.targets)

    # # get the class labels for each image
    # class_labels = [classes[lab] for lab in labels]

    # # log embeddings
    # features = images.view(-1, 28 * 28)
    # writer.add_embedding(features,
    #                     metadata=class_labels,
    #                     label_img=images.unsqueeze(1))


    # Model
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps)
    # print(net)                                                                  ############
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        global global_step
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
    step = 0
    min_loss = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):

    
    
        # for idx, data in enumerate(data_loader):
                        
        #     if min_loss > loss:
        #         best_model = model
    

        loss = train(epoch, net, trainloader, device, optimizer, scheduler,             ########### loss = 
              loss_fn, args.max_grad_norm)
        test(epoch, net, testloader, device, loss_fn, args.num_samples)
                                                                                    ###################
        writer.add_scalar('Traning Loss', loss, global_step=global_step)
        # features = trainloader.view(-1,3 * 28 * 28)
        
        # writer.add_embedding('Images', features)                   ###################
        # running_traning_acc = 
        # step += 1
        # writer.add_scalar('Traning Accuracy ', )

        # writer.add_scalar('Loss/train', np.random.random(), epoch)
        # writer.add_scalar('Loss/test', np.random.random(), epoch)
        # writer.add_scalar('Accuracy/train', np.random.random(), epoch)
        # writer.add_scalar('Accuracy/test', np.random.random(), epoch)
    # torch.save(best_model.state_dict(), 'model.pt')


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    running_loss = 0
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:

            # create grid of images
            dataiter = iter(trainloader)
            images, labels = dataiter.next()
            img_grid = torchvision.utils.make_grid(x)

            # show images
            # matplotlib_imshow(img_grid, one_channel=True)

            # write to tensorboard
            writer.add_image('four_cifar10_images', img_grid)



            writer.add_graph(net, x)
            
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            # if len(x) % 1000 == 999 :                                   # every 1000
            #     progress_bar.set_postfix(nll=loss_meter.avg,
            #                              bpd=util.bits_per_dim(x, loss_meter.avg),
            #                              lr=optimizer.param_groups[0]['lr'])
            #     progress_bar.update(x.size(0))

            global_step += x.size(0)


            # torch.save({
            # 'epoch': EPOCH,
            # 'model_state_dict': net.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': LOSS,
            # }, PATH

            # running_loss += loss.item()
            # if dataiter % 1000 == 999:    # every 1000 mini-batches...

            #     # ...log the running loss
            #     writer.add_scalar('training loss',
            #                     running_loss / 1000,
            #                     epoch * len(trainloader) + i)

            #     # ...log a Matplotlib Figure showing the model's predictions on a
            #     # random mini-batch
            #     writer.add_figure('predictions vs. actuals',
            #                     plot_classes_preds(net, inputs, labels),
            #                     global_step=epoch * len(trainloader) + i)
            #     running_loss = 0.0

            return loss                                                                       #####################


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

@torch.no_grad()
def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


@torch.no_grad()
def test(epoch, net, testloader, device, loss_fn, num_samples):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            # progress_bar.set_postfix(nll=loss_meter.avg,
            #                          bpd=util.bits_per_dim(x, loss_meter.avg))

            # progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InvCon on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')     ## after training   and best result 
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=256, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=25, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=20000, type=int, help='Number of steps for lr warm-up')
    # parser.add_argument('--path', type=str2bool, default='run', help='path to saved model or tensorboard files/ summeryWriter' )
    best_loss = 0
    global_step = 0

    main(parser.parse_args())
    # writer.flush()                                            #######
