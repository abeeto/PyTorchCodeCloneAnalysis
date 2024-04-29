import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from model import CNN
from utils import RunBuilder
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120)


def train(run):
    print('Getting the training data..')
    # downloading FashionMNIST if not already downloaded and putting it into data folder (E and T)
    train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # start the training
    print('Starting the training process on {} examples..'.format(len(train_set)))

    print('\t', run)
    # init the model: we can tune the model hyperparams by adding them to the run and passing them to the constructor
    cnn = CNN()
    # init the optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=run.lr)
    # load the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=True)
    # tensorboard session
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    tb = SummaryWriter(comment=f'{run}')
    tb.add_image('images', grid)
    tb.add_graph(cnn, images)

    for epoch in range(run.num_epochs):
        total_loss = 0
        total_correct = 0  # correctly predicted images
        accuracy = 0

        for batch in train_loader:
            # get training batch from loader
            images, labels = batch

            # pass the batch through the network
            preds = cnn(images)
            # compute the loss
            loss = F.cross_entropy(preds, labels)

            # reinit the grads to avoid cumulativeness
            optimizer.zero_grad()
            # compute the gradients wrt weights
            loss.backward()
            # update the weights
            optimizer.step()

            total_loss += loss.item()
            total_correct += F.softmax(preds, dim=1).argmax(dim=1).eq(labels).sum().item()

        total_loss /= len(train_set)
        accuracy = total_correct / len(train_set)

        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number correct', total_correct, epoch)
        tb.add_scalar('accuracy', accuracy, epoch)

        print('epoch: {}, total_correct: {}, loss: {}, accuracy: {}'.format(epoch, total_correct, total_loss, accuracy))

    tb.close()
    print('Run ended..')
    print('Saving model to disk..')
    cnn.save(run)
    print('Training ended..')


def test(run):
    print('Testing ' + str(run))
    # getting the test data
    test_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    # load the dataset
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    # loading the specified model
    cnn = CNN()
    cnn.load(run)

    test_loss = 0
    test_total_correct = 0
    test_accuracy = 0

    for batch in test_loader:
        images, labels = batch
        preds = cnn(images)
        loss = F.cross_entropy(preds, labels)

        test_loss += loss.item()
        test_total_correct += F.softmax(preds, dim=1).argmax(dim=1).eq(labels).sum().item()

    test_loss /= len(test_set)
    test_accuracy = test_total_correct / len(test_set)

    print('Testing ended.. total_correct: {}, loss: {}, accuracy: {}'.format(test_total_correct, test_loss, test_accuracy))


if __name__ == '__main__':
    # creating the hyperparams
    params = OrderedDict(lr=[0.01, 0.05], batch_size=[100, 1000], num_epochs=[10])

    # building runs from hyperparams (cartesian product of all params)
    runs = RunBuilder.get_runs(params)

    # train and test each run
    for run in runs:
        train(run)
        test(run)
