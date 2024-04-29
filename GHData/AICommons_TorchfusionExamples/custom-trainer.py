from torchfusion.layers import *
from torchfusion.metrics import *
from torchfusion.datasets import *
import torch.nn as nn
import torch.cuda as cuda
from torchfusion.learners import *
from torch.optim import Adam
from torchfusion.utils import clip_grads
from torch.autograd import Variable

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



class CustomLearner(StandardLearner):
    def __train_func__(self, data):

        self.optimizer.zero_grad()

        if self.clip_grads is not None:
            clip_grads(self.model,self.clip_grads[0],self.clip_grads[1])

        train_x, train_y = data

        batch_size = train_x.size(0)

        train_x = Variable(train_x.cuda() if self.cuda else train_x)

        train_y = Variable(train_y.cuda() if self.cuda else train_y)

        outputs = self.model(train_x)
        loss = self.loss_fn(outputs, train_y)
        loss.backward()

        self.optimizer.step()

        self.train_running_loss.add_(loss.cpu() * batch_size)

        for metric in self.train_metrics:
            metric.update(outputs, train_y)

    def __eval_function__(self, data):

        test_x, test_y = data

        test_x = Variable(test_x.cuda() if self.cuda else test_x)

        test_y = Variable(test_y.cuda() if self.cuda else test_y)

        outputs = self.model(test_x)

        for metric in self.test_metrics:
            metric.update(outputs, test_y)

    def __val_function__(self, data):

        val_x, val_y = data
        val_x = Variable(val_x.cuda() if self.cuda else val_x)

        val_y = Variable(val_y.cuda() if self.cuda else val_y)

        outputs = self.model(val_x)

        for metric in self.val_metrics:
            metric.update(outputs, val_y)

    def __predict_func__(self, inputs):

        inputs = Variable(inputs.cuda() if self.cuda else inputs)

        return self.model(inputs)

loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    learner = CustomLearner(model)
    print(learner.summary((1,28,28)))
    learner.train(fmnist_train,loss_fn,optimizer=optimizer,train_metrics=train_metrics,test_loader=fmnist_test,test_metrics=test_metrics,num_epochs=25,model_dir="./fashion-mnist-custommodel")