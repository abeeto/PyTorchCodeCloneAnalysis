class Trainer(object):

    def __init__(self):
        self._num_epochs = 0
        self._optimizer = None
        self._train_loader = None
        self._net = None
        self._print_period = 2000

    def set_net(self, net):
        self._net = net
        return self

    def set_num_epochs(self, num_epochs:int):
        self._num_epochs = num_epochs
        return self

    def set_train_loader(self, train_loader):
        self._train_loader = train_loader
        return self

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        return self

    def set_criterion(self, criterion):
        self._criterion = criterion
        return self

    def train(self):
        for epoch in range(self._num_epochs):
            self._train_one_epoch(epoch)
        print("Finished Training")

    def _train_one_epoch(self, epoch:int):
        running_loss = 0.0
        for i, data in enumerate(self._train_loader):
            inputs, labels = data
            self._optimizer.zero_grad()
            outputs = self._net(inputs)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimizer.step()
            running_loss += loss.item()
            if i % self._print_period == self._print_period - 1:
                print("[%d, %5d] loss: %.3f"%(epoch, i , running_loss/self._print_period))
                running_loss = 0.0

