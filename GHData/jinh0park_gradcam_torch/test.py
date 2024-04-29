from model import ZeroNet
import torch
from data_utils import STL10Loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ZeroNet(num_classes=10)
PATH = 'saved_model'
model.load_state_dict(torch.load(PATH, map_location=device))

stl10 = STL10Loader()
test_loader = stl10.get_loader('test')

with torch.no_grad():
    model = model.to(device)
    model.eval()
    cnt = 0
    correct = 0
    for batch_index, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        scores = model(X)
        predict = scores.argmax(dim=-1)
        correct += predict.eq(y.view_as(predict)).cpu().sum()
        cnt += predict.size(0)

        test_progress = 'Progress: [{}/{} ({:.0f}%)]'.format(
        (batch_index+1), len(test_loader), 100. * (batch_index+1) / len(test_loader))
        print(test_progress, end='\r')
    print()

    test_acc = correct.cpu().item()/cnt*100
    print("Test accuracy: {:.2f}%".format(test_acc))
