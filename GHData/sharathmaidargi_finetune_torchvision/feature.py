import torchvision.models
import sklearn.svm
import torch.nn as nn
import numpy as np
import torch.utils.data
import  matplotlib.pyplot as plt
import torchvision.transforms as transforms
from inception import inception_v3
from util import plot_confusion_matrix


model = inception_v3(pretrained=True)
# model.aux_logit = False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize = transforms.Resize((299, 299))

preprocessor = transforms.Compose([
    resize,
    transforms.ToTensor(),
    normalize,
])


# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, 9)

# new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
# model.classifier = new_classifier

data_dir1 = "./datasets/household/train"
batch_size1 = 64
data_dir2 = "./datasets/household/test"
batch_size2 = 64

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(data_dir1, preprocessor),
    batch_size=batch_size1,
    shuffle=True)


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(data_dir2, preprocessor),
    batch_size=batch_size2,
    shuffle=True)

x_train = None
y_train = None
x_test = None
y_test = None

for i, (in_data, target) in enumerate(train_loader):
    # print "train" +str(i)
    input_var = torch.autograd.Variable(in_data)
    # print "done input var"
    target_var = (torch.autograd.Variable(target))
    # print "done target_var"
    output = (model(input_var))
    del input_var
    # print "done output"
    # print output
    # convert the output of feature extractor to numpy array
    # print "Got output of size = " + str(output.shape)
    # print "Got aux output of size = " + str(aux.shape)
    if x_train is None:
        x_train = output.detach().numpy()
    else:
        x_train = np.append(x_train, output.detach().numpy(), axis=0)
    del output
    if y_train is None:
        y_train = target_var.detach().numpy()
    else:
        y_train = np.append(y_train, target_var.detach().numpy(), axis=0)
    # print "x_train shape = " + str(x_train.shape) + "\ny_train shape = " + str(y_train.shape)
    del target_var

for i, (in_data, target) in enumerate(test_loader):
    # print "test" +str(i)
    input_var = torch.autograd.Variable(in_data)
    # print "done input var"
    target_var = (torch.autograd.Variable(target))
    # print "done target_var"
    output = (model(input_var))
    del input_var
    # print "done output"
    # print output
    # convert the output of feature extractor to numpy array
    # print "Got output of size = " + str(output.shape)
    if x_test is None:
        x_test = output.detach().numpy()
    else:
        x_test = np.append(x_test, output.detach().numpy(), axis=0)
    del output
    if y_test is None:
        y_test = target_var.detach().numpy()
    else:
        y_test = np.append(y_test, target_var.detach().numpy(), axis=0)
    print "x_test shape = " + str(x_test.shape) + "\ny_test shape = " + str(y_test.shape)
    del target_var

gaussian_model = sklearn.svm.SVC(C=1.0, kernel='linear')
loss = gaussian_model.fit(x_train, y_train)
y_pred = gaussian_model.predict(x_test)
accuracy = gaussian_model.score(x_test, y_test)

print "Scikit SVM for Inception feature extractor Dataset : " + str(accuracy.round(4)) + "\n\n"

plot_confusion_matrix(y_pred, y_test)
plt.savefig("conf_feature.png")
plt.show()




