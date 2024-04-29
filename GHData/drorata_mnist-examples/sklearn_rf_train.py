import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

batch_size = 100       # The size of input data took for one iteration

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

print('Loading data')
X = []
y = []
for images, labels in train_loader:
    for img, label in zip(images, labels):
        X.append(img.numpy()[0].reshape(-1))
        y.append(label)

print('Training model...')
pipeline = Pipeline(
    [
        ('scale', StandardScaler()),
        ('clf', RandomForestClassifier(n_jobs=-1))
    ]
)

pipeline.fit(X, y)
print('Training model... DONE')

print('Persisting model...')
joblib.dump(pipeline, './models/sklearn_rf.pkl')
print('Persisting model... DONE')
