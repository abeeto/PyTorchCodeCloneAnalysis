from main import ClassifierPipeline
import argparse
from models.models import *
from data.data import *
import os
import numpy as np
from collections import defaultdict
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--model', type=str, default='SimpleModel', help="SimpleModel or AVModel,..")
parser.add_argument('--dataset', type=str, default='cifar_10', help="cifar_10 or mnist,..")
parser.add_argument('--learning-rate', type=float, default=0.005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--perc-size', type=float, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log-interval', type=int, default=5)
parser.add_argument('--save-interval', type=int, default=6)
parser.add_argument('--use-wandb', action='store_true')
parser.add_argument('--resume-from-saved', type=str, default=None, help="name of the exp to load from")
parser.add_argument('--save-as', type=str, default='', help="a name for the model save file")
args = parser.parse_args()


# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 28*28
BATCH_SIZE = 128

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
USE_PRETRAINED = False
NUM_EPOCHS = 50000
LEARNING_RATE = 0.01
SHAPE = [(IMAGE_SIZE, 1024), (1024, 1024), (1024, 2)]
MARGIN = 5.0



class SiameseModel(nn.Module):
    
    def __init__(self):
        super(SiameseModel, self).__init__()
        image_size = 28*28
        embed_size = 5 # 2
        self.HiddenLayer_1 = nn.Linear(image_size, 1024)
        self.HiddenLayer_2 = nn.Linear(1024, 1024)
        self.OutputLayer = nn.Linear(1024, embed_size)
        
    def forward_once(self, X):
        output = nn.functional.relu(self.HiddenLayer_1(X))
        output = nn.functional.relu(self.HiddenLayer_2(output))
        output = self.OutputLayer(output)
        return output
    
    def forward(self, X1, X2):
        out_1 = self.forward_once(X1)
        out_2 = self.forward_once(X2)
        return out_1, out_2

class SiamesePipeline(ClassifierPipeline):
    def __init__(self, args=None, net=None, datatuple=None):
        super(SiamesePipeline, self).__init__(args=args, net=net, datatuple=datatuple)
        self.args=args
        
        if self.args.use_wandb:
            wandb.init(project="torch-cnn", entity="joeljosephjin", config=args)
        
    def get_contrast_loss(self, out_1, out_2, Y):
        margin = 5.0
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2)
        loss_contrastive = torch.mean((Y) * torch.pow(euclidean_distance, 2) +
                                      (1 - Y) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive, euclidean_distance
        
    def generate_batch(self, dataloader):
        train_iter = iter(dataloader)
        input_1, label_1 = next(train_iter)
        input_2, label_2 = next(train_iter)
        input_1 = input_1.reshape(input_1.size()[0], -1)
        input_2 = input_2.reshape(input_2.size()[0], -1)
        np_label_1 = label_1.numpy()
        np_label_2 = label_2.numpy()
        label = (np_label_1 == np_label_2).astype('float32')
        return input_1, input_2, label
    
    def generate_trainbatch(self):
        return self.generate_batch(self.trainloader)
    
    def generate_testbatch(self):
        return self.generate_batch(self.testloader)
    
    def get_class_embeddings(self):
        # run for few epochs and collect into each class
        self.class_embds = defaultdict(list)
        for i, data in enumerate(self.trainloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.net.forward_once(inputs.reshape(inputs.size()[0], -1)).cpu().detach()
            for i, label in enumerate(labels):
                self.class_embds[label.cpu().item()].append(outputs[i])
                
        for clas in self.class_embds.keys():
            self.class_embds[clas] = sum(self.class_embds[clas])/len(self.class_embds[clas])
                
        print('Class Embeddings Stored Successfully...')
        
    def get_closest_class(self, outputs):
        # compare outputs embeddings with each class embedding and select closes one as predicted
        predicted = []
        for output in outputs:
            distances = []
            for clas in sorted(self.class_embds.keys()):
                distances.append(nn.functional.pairwise_distance(self.class_embds[clas], output))
            
            predicted.append(distances.index(min(distances)))
            
        return torch.tensor(predicted)
        
    def train(self):
        for epoch in range(self.args.epochs):
            input_1, input_2, out = self.generate_trainbatch()
            X_1 = torch.Tensor(input_1).float().to(self.device)
            X_2 = torch.Tensor(input_2).float().to(self.device)
            Y = torch.Tensor(out).float().to(self.device)
            self.optimizer.zero_grad()
            out_1, out_2 = self.net.forward(X_1, X_2)
            loss_val, _ = self.get_contrast_loss(out_1, out_2, Y)
            loss_val.backward()
            self.optimizer.step()
            if epoch % 2 == 0:
                print('Epoch: %d Loss: %.3f' % (epoch, loss_val))
                self.scheduler.step()
            if self.args.use_wandb:
                wandb.log({'loss':loss_val.item()})
            if epoch % 10 == 0:
                self.test()
        
    def test(self):
        self.get_class_embeddings()
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(self.device), data[1]
                # calculate outputs by running images through the network
                outputs = self.net.forward_once(inputs.reshape(inputs.size()[0], -1)).cpu().detach()
                # the class with the highest energy is what we choose as prediction
                predicted = self.get_closest_class(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct // total
        print(f'Accuracy of the network on the 10000 test images: {test_accuracy} %')
        if self.args.use_wandb:
            wandb.log({'valid_acc': test_accuracy})
        
if __name__ == "__main__":
    net = SiameseModel
    datatuple = load_mnist(batch_size=args.batch_size, perc_size=args.perc_size)
    pipeline = SiamesePipeline(args=args, net=net, datatuple=datatuple)
    pipeline.train()
    pipeline.test()
