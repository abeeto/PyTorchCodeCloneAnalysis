from net_8 import *

"""
================================================================
================================================================
========================== Function ============================
================================================================
"""


def net8TrainOnce(net, trn_loader, loss_print):
    running_loss = 0.0
    for i, (input, label) in enumerate(trn_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input)
        # print(outputs.is_cuda())
        label = label.to(device, dtype=torch.long)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
            loss_print.append(running_loss / 20)
            running_loss = 0.0
    return net, loss_print


# Train Model
def train_cnn(net, train_data, train_label):
    loss_print = []
    for e in range(epoch):  # loop over the dataset multiple times
        print('Start Training epoch {}, batch size {}, remaining {} epochs'.format(e + 1, batch_size, epoch - e - 1))

        # split train data and test data
        trn_loader, tst_loader = split_trn_tst(train_data, train_label, batch_size=batch_size, trans=True, shuff=True,
                                               whole=True)
        # Train once
        net, loss_print = net8TrainOnce(net, trn_loader, loss_print)

        # test mode;
        net8TestOnce(net, tst_loader)

        # save model every epoch
        torch.save(net.state_dict(), "{}/epoch{}.pth".format("./checkpoint", e))

    plotTrain(epoch * 3, loss_print)
    print('Finished Training')

    return net


"""
================================================================
============== Load Data and Train Model =======================
================================================================
"""

# TODO just run this file, will train the model
if __name__ == '__main__':
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ('a', 'b', 'c', 'd', 'h', 'i', 'j', 'k')

    net = Net_8().to(device)  # init net
    batch_size = 64
    epoch = 450

    # define loss function
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)  # .to(device)

    # Load Train Data
    pkl_file = "./train_data.pkl"
    npy_file = "./finalLabelsTrain.npy"
    data, label = loadFile(pkl_file, npy_file)
    data = preprocessor(data, 52, 52)

    train_data = data[0:6400 - 640]
    train_label = label[0:6400 - 640]

    test_data = data[6400 - 640:6400]
    test_label = label[6400 - 640:6400]

    # train model
    net = train_cnn(net, train_data, train_label)
