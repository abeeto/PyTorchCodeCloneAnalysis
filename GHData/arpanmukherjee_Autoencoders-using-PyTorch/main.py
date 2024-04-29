import os
import torch
import warnings
import argparse
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Autoencoder, ConvolutionAE


def main():
    parser = argparse.ArgumentParser(
        description='Simple training script for training model')

    parser.add_argument(
        '--epochs', help='Number of epochs (default: 75)', type=int, default=75)
    parser.add_argument(
        '--batch-size', help='Batch size of the data (default: 16)', type=int, default=16)
    parser.add_argument(
        '--learning-rate', help='Learning rate (default: 0.001)', type=float, default=0.001)
    parser.add_argument(
        '--seed', help='Random seed (default:1)', type=int, default=1)
    parser.add_argument(
        '--data-path', help='Path for the downloaded dataset (default: ../dataset/)', default='../dataset/')
    parser.add_argument(
        '--dataset', help='Dataset name. Must be one of MNIST, STL10, CIFAR10')
    parser.add_argument(
        '--use-cuda', help='CUDA usage (default: False)', type=bool, default=False)
    parser.add_argument(
        '--network-type', help='Type of the network layers. Must be one of Conv, FC (default: FC)', default='FC')
    parser.add_argument(
        '--weight-decay', help='weight decay (L2 penalty) (default: 1e-5)', type=float, default=1e-5)
    parser.add_argument(
        '--log-interval', help='No of batches to wait before logging training status (default: 50)', type=int, default=50)
    parser.add_argument(
        '--save-model', help='For saving the current model (default: True)', type=bool, default=True)

    args = parser.parse_args()

    epochs = args.epochs  # number of epochs
    batch_size = args.batch_size  # batch size
    learning_rate = args.learning_rate  # learning rate
    torch.manual_seed(args.seed)  # seed value

    # Creating dataset path if it doesn't exist
    if args.data_path is None:
        raise ValueError('Must provide dataset path')
    else:
        data_path = args.data_path
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

    # Downloading proper dataset and creating data loader
    if args.dataset == 'MNIST':
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_data = torchvision.datasets.MNIST(
            data_path, train=True, download=True, transform=T)
        test_data = torchvision.datasets.MNIST(
            data_path, train=False, download=True, transform=T)

        ip_dim = 1 * 28 * 28  # input dimension
        h1_dim = int(ip_dim / 2)  # hidden layer 1 dimension
        op_dim = int(ip_dim / 4)  # output dimension
    elif args.dataset == 'STL10':
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = torchvision.datasets.STL10(
            data_path, split='train', download=True, transform=T)
        test_data = torchvision.datasets.STL10(
            data_path, split='test', download=True, transform=T)

        ip_dim = 3 * 96 * 96  # input dimension
        h1_dim = int(ip_dim / 2)  # hidden layer 1 dimension
        op_dim = int(ip_dim / 4)  # output dimension
    elif args.dataset == 'CIFAR10':
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = torchvision.datasets.CIFAR10(
            data_path, train=True, download=True, transform=T)
        test_data = torchvision.datasets.CIFAR10(
            data_path, train=False, download=True, transform=T)

        ip_dim = 3 * 32 * 32  # input dimension
        h1_dim = int(ip_dim / 2)  # hidden layer 1 dimension
        op_dim = int(ip_dim / 4)  # output dimension
    elif args.dataset is None:
        raise ValueError('Must provide dataset')
    else:
        raise ValueError('Dataset name must be MNIST, STL10 or CIFAR10')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # use CUDA or not
    device = 'cpu'
    if args.use_cuda is False:
        if torch.cuda.is_available():
            warnings.warn(
                'CUDA is available, please use for faster convergence')
        else:
            device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            raise ValueError('CUDA is not available, please set it False')

    # Type of layer
    if args.network_type == 'FC':
        auto_encoder = Autoencoder(ip_dim, h1_dim, op_dim).to(device)
    elif args.network_type == 'Conv':
        auto_encoder = ConvolutionAE().to(device)
    else:
        raise ValueError('Network type must be either FC or Conv type')

    # Train the model
    auto_encoder.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        lr=learning_rate, params=auto_encoder.parameters(), weight_decay=args.weight_decay)

    for n_epoch in range(epochs):  # loop over the dataset multiple times
        reconstruction_loss = 0.0
        for batch_idx, (X, Y) in enumerate(train_loader):
            X = X.view(X.size()[0], -1)
            X = Variable(X).to(device)

            encoded, decoded = auto_encoder(X)

            optimizer.zero_grad()
            loss = criterion(X, decoded)
            loss.backward()
            optimizer.step()

            reconstruction_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                print('[%d, %5d] Reconstruction loss: %.5f' %
                      (n_epoch + 1, batch_idx + 1, reconstruction_loss / args.log_interval))
                reconstruction_loss = 0.0
    if args.save_model:
        torch.save(auto_encoder.state_dict(), "Autoencoder.pth")

    # Save real images
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    torchvision.utils.save_image(torchvision.utils.make_grid(
        images, nrow=4), 'images/actual_img.jpeg')

    # Load trained model and get decoded images
    auto_encoder.load_state_dict(torch.load('Autoencoder.pth'))
    auto_encoder.eval()
    images = images.view(images.size()[0], -1)
    images = Variable(images).to(device)
    encoded, decoded = auto_encoder(images)

    # Save decoded images
    if args.dataset == 'MNIST':
        decoded = decoded.view(decoded.size()[0], 1, 28, 28)
    elif args.dataset == 'STL10':
        decoded = decoded.view(decoded.size()[0], 3, 96, 96)
    elif args.dataset == 'CIFAR10':
        decoded = decoded.view(decoded.size()[0], 3, 32, 32)
    torchvision.utils.save_image(torchvision.utils.make_grid(
        decoded, nrow=4), 'images/decoded_img.jpeg')


if __name__ == '__main__':
    main()
