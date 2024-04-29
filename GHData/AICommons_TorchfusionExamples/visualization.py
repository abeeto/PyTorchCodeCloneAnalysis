from torchfusion.layers import *
from torchfusion.metrics import *
from torchfusion.datasets import *
import torch.nn as nn
import torch.cuda as cuda
from torchfusion.learners import *
from torch.optim import Adam
from torchfusion.utils import VisdomLogger

model = nn.Sequential(
    Flatten(),
    Linear(784, 100),
    Swish(),
    Linear(100, 100),
    Swish(),
    Linear(100, 100),
    Swish(),
    Linear(100, 100),
    Swish(),
    Linear(100, 10)
)

if cuda.is_available():
    model.cuda()

optimizer = Adam(model.parameters())

fmnist_train = fashionmnist_loader(28,batch_size=128)
fmnist_test = fashionmnist_loader(28,batch_size=128,train=False)

train_metrics = [Accuracy()]
test_metrics = [Accuracy()]

loss_fn = nn.CrossEntropyLoss()
visdom_loggger = VisdomLogger()

if __name__ == "__main__":
    learner = StandardLearner(model)
    print(learner.summary((1,28,28)))
    learner.train(fmnist_train,loss_fn,optimizer=optimizer,train_metrics=train_metrics,test_loader=fmnist_test,test_metrics=test_metrics,tensorboard_log="./tboard_logs",visdom_log=visdom_loggger,num_epochs=25,model_dir="./fashion-mnist-model")