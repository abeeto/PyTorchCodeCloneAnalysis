import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.autograd import Variable
import torch.optim as optim

from data_preprocess import CustomDataset
from loss_criterion import CrossEntropyLoss2d
from seg_net import Net
from tranform import *


device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")


def plot_loss(loss_values, epoch):
    epoch_values = np.arange(1, epoch + 2)
    plt.plot(epoch_values, loss_values)
    plt.show()


def main(params):

    input_transform = Compose([
        Resize((params["resize_width"], params["resize_width"])),
        ToTensor(),
        # TODO: normalize according to training data statistics
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    target_transform = Compose([
        Resize((params["resize_width"], params["resize_width"])),
        ToLabel(),
        Relabel(params["object_pixel"], params["number_classes"] - 1),
    ])

    train_dataset = CustomDataset(params["image_dir"], params["mask_dir"], input_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False,
                                               num_workers=params["num_workers"])

    net = Net()
    criterion = CrossEntropyLoss2d()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_values = []
    for epoch in range(params["epochs"]):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = net(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss_values.append(loss.item())
            print('loss: ', loss)
            loss.backward()
            optimizer.step()

            checkpoint = {'model': net,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}

        if epoch % params["save_steps"] == 0:
            torch.save(checkpoint, params["checkpoint"])
            print("filename: {} for epoch {}".format(params["checkpoint"], epoch))

    plot_loss(loss_values, epoch)


if __name__ == '__main__':

    params = {

        "image_dir": os.path.join(os.getcwd(), 'train_images'),
        "mask_dir":  os.path.join(os.getcwd(), 'mask'),
        "number_classes": 2,
        "object_pixel": 255,
        "batch_size": 1,
        "epochs": 600,
        "save_steps": 10,
        "num_workers": 0,
        "resize_width": 256,
        "resize_height": 256,
        "checkpoint": os.path.join(os.getcwd(), "weights/checkpoint.pth")
    }
    main(params)