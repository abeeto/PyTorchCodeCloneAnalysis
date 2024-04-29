from main import ClassifierPipeline
import argparse
from models.models import *
from data.data import *
import os
import numpy as np
from collections import defaultdict
import wandb
from utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model', type=str, default='SimpleModel', help="SimpleModel or AVModel,..")
    parser.add_argument('--dataset', type=str, default='cifar_10', help="cifar_10 or mnist,..")
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--perc-size', type=float, default=1)
    parser.add_argument('--n-shot', type=float, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=6)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--resume-from-saved', type=str, default=None, help="name of the exp to load from")
    parser.add_argument('--save-as', type=str, default='', help="a name for the model save file")
    args = parser.parse_args()
    
    return args


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
            wandb.init(project="torch-cnn", entity="joeljosephjin", config=self.args)
        
    def get_contrast_loss(self, out_1, out_2, Y):
        margin = 5.0
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2)
        loss_contrastive = torch.mean((Y) * torch.pow(euclidean_distance, 2) +
                                      (1 - Y) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive, euclidean_distance
        
    def generate_batch(self, dataloader):
        # filter in only specific classes
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
    
    def get_class_embeddings(self, n_shot=10, test_classes=[0, 1, 2]):
        """
        NOT EFFICIENT
        run few batches and collect into each class n samples
        and then stop storing once n samples are done
        once all classes have n samples, finish the loop
        EFFICIENT (but NOT POSSIBLE without datasets module)
        select indices of samples of each class
        run a loop n times
        each time store all the required classes embeddings
        by sampling indices randomly
        """
        self.class_embds = defaultdict(list)
        dataloader = self.testloader
        # select indices of classes
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.net.forward_once(inputs.reshape(inputs.size()[0], -1)).cpu().detach()
            for i, label in enumerate(labels):
                if len(self.class_embds[label.cpu().item()]) >= n_shot:
                    continue
                self.class_embds[label.cpu().item()].append(outputs[i])
                
            # if all the classes are filled, then break the loop
            if all([len(self.class_embds[x])>=n_shot for x in test_classes]):
                break
                
        for clas in self.class_embds.keys():
            self.class_embds[clas] = sum(self.class_embds[clas])/len(self.class_embds[clas])
                
        print(f'Class Embeddings Stored Successfully for classes {list(self.class_embds.keys())}...')
        
    def get_closest_class(self, outputs, classes=None):
        # compare outputs embeddings with each class embedding and select closes one as predicted
        predicted = []
        for output in outputs:
            distances = []
            for clas in sorted(self.class_embds.keys()):
                distances.append(nn.functional.pairwise_distance(self.class_embds[clas], output))
            
            if classes:
                predicted.append(classes[distances.index(min(distances))])
            else:
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
        self.get_class_embeddings(n_shot=self.args.n_shot, test_classes=self.args.test_classes)
        correct = 0
        total = 0
        
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.args.test_classes}
        total_pred = {classname: 0 for classname in self.args.test_classes}
        
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(self.device), data[1]
                # calculate outputs by running images through the network
                outputs = self.net.forward_once(inputs.reshape(inputs.size()[0], -1)).cpu().detach()
                # the class with the highest energy is what we choose as prediction
                predictions = self.get_closest_class(outputs, self.args.test_classes)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[label.item()] += 1
                    total_pred[label.item()] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Test Accuracy for class: {classname} is {accuracy:.1f} %')
            
        test_accuracy = 100 * sum(correct_pred.values()) // sum(total_pred.values())
        print(f'Overall Test Accuracy for {sum(total_pred.values())} samples: {test_accuracy} %')
        if self.args.use_wandb:
            wandb.log({'valid_acc': test_accuracy})
        
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    net = SiameseModel
    # datatuple = load_mnist(batch_size=args.batch_size, perc_size=args.perc_size)
    # test_classes = [0, 1, 2]
    test_classes = [0, 2]
    args.test_classes = test_classes
    datatuple = load_fewshot_mnist(batch_size=args.batch_size, perc_size=1, test_classes=test_classes)
    pipeline = SiamesePipeline(args=args, net=net, datatuple=datatuple)
    pipeline.train()
    pipeline.test()
