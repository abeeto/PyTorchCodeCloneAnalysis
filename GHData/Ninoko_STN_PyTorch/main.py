import torch
import torch.optim as optim

from data.dataloader import mnist_loader
from architectures.models import LeNet
from src.activations import GELU
from src.utils import weights_init
# from torch.nn.modules.loss import CrossEntropyLoss
from src.losses import CrossEntropyLoss
from src.callbacks import *

NUM_EPOCH = 10
training_loss_id = 0
test_loss_id = 0


def train(model, criterion, optimizer, device='cpu'):
    global training_loss_id
    model.train()
    loader = mnist_loader['train']

    img_num = train_loss = batch_idx = 0

    for batch_idx, (img, label) in enumerate(loader):
        img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        training_loss_id, train_loss, img_num = after_batch(training_loss_id, img, loss,
                                                            train_loss, img_num, mode='Training')
    after_epoch(epoch_id, train_loss, img_num, batch_idx)


def test(model, criterion, device='cpu'):
    global test_loss_id
    model.eval()
    loader = mnist_loader['test']

    img_num = test_loss = batch_idx = 0

    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(loader):
            img, mask = img.to(device, dtype=torch.float), mask.to(device, dtype=torch.long)
            output = model(img)
            loss = criterion(output, mask)
            test_loss_id, test_loss, img_num = after_batch(test_loss_id, img, loss,
                                                           test_loss, img_num, mode='Validation')
    after_epoch(epoch_id, test_loss, img_num, batch_idx)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LeNet(activation=GELU()).to(device)
    model.apply(weights_init)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=0.0005)
    criterion = CrossEntropyLoss().to(device)

    for epoch_id in range(NUM_EPOCH):
        print("Epoch:\t{}".format(epoch_id))
        train(model, criterion, optimizer, device=device)
        test(model, criterion, device=device)
