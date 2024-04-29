import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from sklearn.metrics import classification_report

from sklearn.externals import joblib

batch_size = 100       # The size of input data took for one iteration

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
print('Loading data')
X = []
y = []
for images, labels in train_loader:
    for img, label in zip(images, labels):
        X.append(img.numpy()[0].reshape(-1))
        y.append(label)

print('Loading persisted model...')
clf = joblib.load('./models/sklearn_rf.pkl')
print('Starting predictions...')
y_pred = clf.predict(X)
print('Starting predictions... DONE')

print(classification_report(y, y_pred))
