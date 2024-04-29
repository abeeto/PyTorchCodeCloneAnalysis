from torchfusion.layers import *
from torchfusion.metrics import *
from torchfusion.datasets import *
import torch.nn as nn
import torch.cuda as cuda
from torchfusion.learners import *
from torch.optim import Adam

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
    model = model.cuda()

optimizer = Adam(model.parameters())

mnist_train = mnist_loader(28,batch_size=128)
mnist_test = mnist_loader(28,batch_size=128,train=False)

train_metrics = [Accuracy()]
test_metrics = [Accuracy()]

loss_fn = nn.CrossEntropyLoss()


if __name__ == "__main__":
    learner = StandardLearner(model)

    print(learner.summary((1,28,28)))
    learner.train(mnist_train,loss_fn,optimizer=optimizer,train_metrics=train_metrics,test_loader=mnist_test,test_metrics=test_metrics,num_epochs=5,model_dir="./mnist-model",save_logs="mnist-logs.txt",save_metrics=True,batch_log=False)
    print(learner.get_train_history())
