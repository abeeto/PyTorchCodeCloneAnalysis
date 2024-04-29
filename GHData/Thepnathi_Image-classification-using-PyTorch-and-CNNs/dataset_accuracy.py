"""
Assignment 2, COMP338 - Step 4.1 Test datasets against the ground-truth. Uses all the trained models

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

from cnn import ConvolutionalNetwork
from constants import Constants, load_trained_models
import torch as th

class Dataset_Accuracy(object):
    def __init__(self, net):
        self.net = net

    def dataset_accuracy(self, dataset, name=""):
        correct_prediction = 0
        total = 0

        for i in range(len(dataset)):
          image = th.tensor([dataset[i]['imNorm']])
          label = th.tensor([dataset[i]['label']])
          image, label = image.to(Constants.device), label.to(Constants.device)
          outputs = self.net(image)
          _, predicted = th.max(outputs, 1)
          total += label.size(0)
          correct_prediction += (predicted == label).sum()

        accuracy = 100 * float(correct_prediction) / total
        print('Accuracy of the network on the {} {} images: {:2f} %'.format(total, name, accuracy))

    def train_dataset_accuracy(self, train_dataset):
        self.dataset_accuracy(train_dataset, "train")

    def validation_dataset_accuracy(self, validation_dataset):
        self.dataset_accuracy(validation_dataset, "validation")

    def test_dataset_accuracy(self, test_dataset):
        self.dataset_accuracy(test_dataset, "test")

    def compute_dataset_accuracy(self, train_dataset=None, test_dataset=None):
        self.train_dataset_accuracy(train_dataset) if train_dataset else None
        self.test_dataset_accuracy(test_dataset) if test_dataset else None
        print("====================")


if __name__ == "__main__":
    # load all the trained models
    trained_models = load_trained_models()

    # Calculates the overall prediction accuracy of the train and test dataset on each of the trained cnn models by learning rate
    for num_epochs in Constants.num_epochs:
        for rate in Constants.learning_rates:
            print(f'Learning rate: {rate}, Number of epochs: {num_epochs}')
            loaded_trained_model = trained_models[rate]
            dataset_accuracy = Dataset_Accuracy(loaded_trained_model)
            dataset_accuracy.compute_dataset_accuracy(Constants.train_dataset, Constants.test_dataset)
