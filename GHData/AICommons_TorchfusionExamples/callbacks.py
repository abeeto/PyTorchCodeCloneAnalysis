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

mnist_train = mnist_loader(28, batch_size=128, root="./mnist")
mnist_test = mnist_loader(28, batch_size=128, train=False, root="./mnist")

train_metrics = [Accuracy()]
test_metrics = [Accuracy()]

loss_fn = nn.CrossEntropyLoss()

def batch_start(epoch_num,batch_num):
    if batch_num % 100 == 0:
        print("Starting batch {} of epoch {}".format(batch_num,epoch_num))

def batch_end(epoch_num,batch_num, batch_info):
    if batch_num % 100 == 0:
        for key in batch_info:
            print("Epoch {} Batch {}, {} : {}".format(epoch_num,batch_num, key, batch_info[key]))


def epoch_start(epoch_num):
    print("Starting epoch {}".format(epoch_num))

def epoch_end(epoch_num, epoch_info):
    for key in epoch_info:
        print("Epoch {} , {} : {}".format(epoch_num, key, epoch_info[key]))

def completed(info):
    for key in info:
        print("{} : {}".format(key,info[key]))


if __name__ == "__main__":
    learner = StandardLearner(model)
    learner.add_on_batch_end(batch_end)
    learner.add_on_batch_start(batch_start)
    learner.add_on_epoch_start(epoch_start)
    learner.add_on_epoch_end(epoch_end)
    learner.add_on_training_completed(completed)

    print(learner.summary((1, 28, 28)))
    learner.train(mnist_train, loss_fn, optimizer=optimizer, train_metrics=train_metrics, test_loader=mnist_test,
                  test_metrics=test_metrics, num_epochs=4, model_dir="./mnist-model")
    print(learner.get_train_history())