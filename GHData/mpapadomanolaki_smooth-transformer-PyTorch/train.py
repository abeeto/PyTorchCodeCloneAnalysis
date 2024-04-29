import nets as N
import utils as U
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt
import shutil
import os

if __name__ == '__main__':

    def train(epochs, model, mov_train_loader, ref_train_loader, optimizer, epoch, criterion):
        model.train()
        #set regularizers for affine and deformable registration
        a, b = 1e-4, 1e-7
        log_interval = 10
        total_loss = 0
        for batch_idx, (data, target) in enumerate(zip(mov_train_loader, ref_train_loader)):
            mov_data, ref_data = data, target
            mov_data[0] = U.to_cuda(torch.stack((mov_data[0],mov_data[0],mov_data[0]), 1).squeeze(2))
            ref_data[0] = U.to_cuda(torch.stack((ref_data[0],ref_data[0],ref_data[0]), 1).squeeze(2))
            optimizer.zero_grad()
            deformable, affine, reg_info = model(mov_data[0], ref_data[0])
            deformed, ref_def, sampling_grid_norm, sampling_grid_inverse = reg_info
            aff_loss = a*torch.sum(torch.abs(affine))
            def_loss = b*torch.sum(torch.abs(deformable))
            loss = criterion(deformed, ref_data[0]) + def_loss + aff_loss 
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(mov_data), len(mov_train_loader.dataset),
                    100. * batch_idx / len(mov_train_loader), loss.item()))
        print('Mean loss: ', total_loss/float(batch_idx)) #float(len(mov_train_loader.dataset)))


    def test(model, mov_test_loader, ref_test_loader, criterion):
        model.eval()
        a, b = 1e-4, 1e-7
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(zip(mov_test_loader, ref_test_loader)):
                mov_data, ref_data = data, target
                mov_data[0] = U.to_cuda(torch.stack((mov_data[0],mov_data[0],mov_data[0]), 1).squeeze(2))
                ref_data[0] = U.to_cuda(torch.stack((ref_data[0],ref_data[0],ref_data[0]), 1).squeeze(2))
                deformable, affine, reg_info = model(mov_data[0], ref_data[0])
                deformed, ref_def, sampling_grid_norm, sampling_grid_inverse = reg_info
                aff_loss = a*torch.sum(torch.abs(affine))
                def_loss = b*torch.sum(torch.abs(deformable))
                test_loss += criterion(deformed, ref_data[0]) + def_loss + aff_loss 

        test_loss = test_loss/float(batch_idx) #len(mov_test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))

    if os.path.exists('data'):
        print('data exist')
    else:
        os.mkdir('data')


    transform=transforms.Compose([
        transforms.ToTensor()
        ])

    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    mov_train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=256)
    mov_test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, batch_size=256)

    ref_train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=256)
    ref_test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, batch_size=256)

    model = U.to_cuda(N.DisplNet(6))
    model = U.initialize_model(model)

    criterion = U.to_cuda(torch.nn.MSELoss())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epochs, model, mov_train_loader, ref_train_loader, optimizer, epoch, criterion)
        test(model, mov_test_loader, ref_test_loader, criterion)

    torch.save(model.state_dict(), "mnist_cnn.pt")
