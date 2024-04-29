import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ZeroNet
from data_utils import STL10Loader
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(data_path='data',batch_size=10, epochs=150, num_classes=10):
    model = ZeroNet(num_classes=num_classes).to(device)

    stl10 = STL10Loader()

    train_loader = stl10.get_loader('train')
    valid_loader = stl10.get_loader('valid')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(epochs):
        scheduler.step()
        model.train()
        for batch_index, (X, y) in enumerate(train_loader):
            model.zero_grad()
            X, y = X.to(device), y.to(device)
            scores = model(X)
            loss = F.cross_entropy(scores, y)

            loss.backward()
            optimizer.step()

            if batch_index % 1 == 0:
                train_log = 'Epoch {:2d}/{:2d}\tLoss: {:.6f}\tTrain: [{}/{} ({:.0f}%)]'.format(
                epoch, epochs, loss.cpu().item(), (batch_index+1), len(train_loader),
                100. * (batch_index+1) / len(train_loader))
                print(train_log, end='\r')
        print()

        correct = 0.
        last_valid_acc = 0.

        model.eval()
        with torch.no_grad():
            cnt = 0
            for batch_index, (X, y) in enumerate(valid_loader):
                X, y = X.to(device), y.to(device)
                scores = model(X)
                predict = scores.argmax(dim=-1)
                correct += predict.eq(y.view_as(predict)).cpu().sum()
                cnt += predict.size(0)
            valid_acc = correct.cpu().item()/cnt*100
            print("validation accuracy: {:.2f}%".format(valid_acc))
            last_valid_acc = valid_acc

    SAVE_PATH = 'saved_model'
    torch.save(model.state_dict(), SAVE_PATH)
    with open('valid.txt', 'w') as f:
        f.write(str(last_valid_acc))



if __name__=="__main__":
    main()
