import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import logging

logging.basicConfig(format='%(levelname)s:%(message)s',
                    filename="client.log", filemode='w',
                    level=logging.INFO)

class client(object):
    def __init__(self,
                 model,
                 rank,
                 world_size,
                 args=None):

        self.create_logger(rank)

        self.rank = rank
        self.world_size = world_size
        self.load_data_local(f"data/data_worker{rank}_")
        self.model = model
        self.optimizer = optim.SGD(model.parameters(),
                                   lr=0.001)

        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.epochs

        self.logger.info(f"Client {self.rank} Initialized")


    def load_global_model(self,global_params):
        self.logger.info(f"Client {self.rank} Loading Global Weights")
        self.model.load_state_dict(global_params)

    def send_local_model(self):
        self.logger.info(f"Client {self.rank} Sending Local Weights")
        return self.model.state_dict()

    def send_num_train(self):
        return self.n_train

    def train(self):
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            # running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # # print statistics
                # running_loss += loss.item()
                # if i % 2000 == 1999:  # print every 2000 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / 2000))
                #     running_loss = 0.0

        self.logger.info('Finished Training')

    def evaluate(self):

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.logger.info(f"Client {self.rank} Evaluating Data: {round(correct/total,3)}")
        return correct,total

    def load_data_local(self,datapath):
        self.trainloader = torch.load(datapath+"train.pt")
        self.testloader = torch.load(datapath+"test.pt")

        self.n_train = len(self.trainloader.dataset)
        self.logger.info("Local Data Statistics:")
        self.logger.info("Dataset Size: {:.2f}".format(self.n_train))
        self.logger.info(dict(Counter(self.trainloader.dataset[:][1].numpy().tolist())))

    def create_logger(self,rank):
        self.logger = logging.getLogger(f'client{rank}')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(f"client{rank}.log",mode='w',encoding='utf-8'))

