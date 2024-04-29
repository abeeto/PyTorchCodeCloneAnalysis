import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
		
		
		
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def metrics(model, data_loader):

  for X, y in data_loader:

    X, y = X.to(device), y.to(device)

    probs = model(X)
    preds = probs.argmax(dim=1)

    correct = (preds == y).sum().item()

    preds = preds.cpu().detach().numpy()
    y = y.cpu().detach().numpy()


    accuracy_val = correct / len(data_loader.dataset)
    precision_val = precision_score(y, preds, average='macro')
    recall_val = recall_score(y, preds, average='macro')
    f1score_val = f1_score(y, preds, average='macro')

    print(f'accuracy: {accuracy_val}, precision: {precision_val}, recall: {recall_val}, f1_score: {f1score_val}')




def loss_fn(pred, y):

  y_true = F.one_hot(y, num_classes = 10)
  y_pred_log = torch.log(pred)
  loss_tensor = -(y_true * y_pred_log).sum()

  return loss_tensor


class MyNN(nn.Module):

  def __init__(self):
    super(MyNN, self).__init__()

    self.fc = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

  def forward(self, x):

    x = nn.Flatten()(x)
    logits = self.fc(x)
    probs = nn.Softmax(dim=1)(logits)

    return probs



class MyCNN(nn.Module):

  def __init__(self):
    super(MyCNN, self).__init__()

    self.conv_layers = nn.Sequential(
        nn.Conv2d(1, 5, 3, 1),
        nn.ReLU(),
        nn.Conv2d(5, 10, 3, 1),
    )

    num_features = 10*24*24

    self.fc = nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

  def forward(self, x):

    feat_maps = self.conv_layers(x)
    features = nn.Flatten()(feat_maps)
    logits = self.fc(features)
    probs = nn.Softmax(dim=1)(logits)

    return probs



class MyCnnDynamic(nn.Module):

    def __init__(self, config, H, W, num_classes):

        super(MyCnnDynamic, self).__init__()

        temp = []

        for k in config:

            in_ch, out_ch, kernel_size, stride, padding = k
            
            temp.append(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=padding)
            )
        
        self.conv_layers = nn.Sequential(*temp)

        def _in_features(H, W, config):

            for k in config:
                in_ch, out_ch, kernel_size, stride, padding = k

                H = H if padding == 'same' else (H - (kernel_size[0]-1) + 2 * padding) / stride
                W = W if padding == 'same' else (W - (kernel_size[1]-1) + 2 * padding) / stride

            return H * W * config[-1][1]
        
        in_features = _in_features(H, W, config)

        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512,  num_classes),
        )


    def forward(self, x):

        feat_maps = self.conv_layers(x)
        feats = nn.Flatten(start_dim=1)(feat_maps)
        logits = self.fc(feats)
        probs = nn.Softmax(dim=1)(logits)

        return probs





def train_dynamic_config(config, H, W, num_classes, train_loader, test_loader):
	
	model = MyCnnDynamic(config, H, W, num_classes)
	model = model.to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

	epochs = 5
	for t in range(epochs):
	    print(f"Epoch {t+1}\n-------------------------------")
	    train(train_loader, model, loss_fn, optimizer)
	    test(test_loader, model, loss_fn)

	print("Done!")
	
	return model





