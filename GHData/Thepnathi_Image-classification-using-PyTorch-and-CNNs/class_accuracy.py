"""
Assignment 2, COMP338 - Step 4.2 Compute and report the overall and classification errors per class
Assignment 2, COMP338 - Step 4.3 Compute and show the confusion matrix and analyze the results

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import numpy as np
import torch as th
import cv2 as cv2
from constants import Constants, Plot_Tools, load_trained_models, display_predictions

class Class_Accuracy(object):
    def __init__(self, net, classes=Constants.CLASSES):
        self.net = net
        self.classes = classes
        self.num_class = len(self.classes)

    def showImg(self, image, predicted):
        window_name = f'Predicted: {Constants.CLASSES[predicted]}'
        cv2.imshow(window_name, image)
        cv2.waitKey(0)

    # Takes quite long to compute for one model. Might be good idea to use cuda for this part
    def compute_confusion_matrix_for_class_accuracy(self, dataset, batch_size=Constants.default_batch_size, show_images=False):
        # confusion matrix (real label, predicted label)
        confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)

        correct_predicted = [[] for _ in range(len(Constants.CLASSES))]
        incorrect_predicted = [[] for _ in range(len(Constants.CLASSES))]
        NUM_TO_SHOW = 5

        for i in range(len(dataset)):
            image, label = th.tensor([dataset[i]['imNorm']]), th.tensor([dataset[i]['label']])
            image, label = image.to(Constants.device), label.to(Constants.device)
            result = self.net(image)
            _, predicted = th.max(result.data, 1)
            # print(Constants.CLASSES[label])
            # self.showImg(dataset[i]['im'], predicted)
            confusion_matrix[label, predicted] += 1

            # Unpack from tensors.
            predicted, label = int(predicted), int(label)
            if predicted == label and len(correct_predicted[predicted]) < NUM_TO_SHOW:
                correct_predicted[predicted].append(dataset[i])
            elif predicted != label and len(incorrect_predicted[predicted]) < NUM_TO_SHOW:
                incorrect_predicted[predicted].append(dataset[i])


        print("{:<10} {:^10}".format("Class", "Accuracy (%)"))

        overall_correct = 0
        for i in range(self.num_class):
            class_total = confusion_matrix[i, :].sum()
            class_correct = confusion_matrix[i, i]
            percentage_correct = 100.0 * float(class_correct) / class_total
            overall_correct += percentage_correct

            print('{:<10} {:^10.2f}'.format(self.classes[i], percentage_correct))

        print(f'\nOverall Accuracy: {overall_correct // self.num_class}%')

        if show_images:
            print(f'\nShowing {NUM_TO_SHOW} correctly and {NUM_TO_SHOW} incorrectly predicted images for each class\n')
            for c_idx, c in enumerate(Constants.CLASSES):
                display_predictions(c, correct_predicted[c_idx], incorrect_predicted[c_idx], title=f"Images predicted as {c}.")

        return confusion_matrix


if __name__ == "__main__":
    # dictionary that contains the trained cnn model, where key is the model's learning rate
    trained_models = load_trained_models()

    # dictionary to stores the confusion matrix of each model, where key is the learning rates
    confusion_matrix_by_learning_rates = {}

    # Iterate through each model, compute and stores the confusion matrix.
    for num_epochs in Constants.num_epochs:
        confusion_matrix_by_learning_rates[num_epochs] = {}
        for rate in Constants.learning_rates:
            print(f'Learning rate: {rate}, Number of epochs: {num_epochs}')

            # Only display images for step 4.4 for the best model.
            show_images = True if num_epochs == 20 and rate == 1e-05 else False

            class_acc = Class_Accuracy(trained_models[rate], Constants.CLASSES)
            confusion_matrix_by_learning_rates[num_epochs][rate] = \
                class_acc.compute_confusion_matrix_for_class_accuracy(Constants.test_dataset, show_images=show_images)

            print(Constants.line + "\n")


    # For each model, use matplotlib to display the confusion matrix
    for num_epochs in Constants.num_epochs:
        for rate in Constants.learning_rates:
            print(confusion_matrix_by_learning_rates[num_epochs][rate])
            print(Constants.line)

            Plot_Tools.plot_confusion_matrix(cm=confusion_matrix_by_learning_rates[num_epochs][rate],
                                            classes=Constants.CLASSES,
                                            normalize=False,
                                            title=f'Confusion matrix of CCN model with learning rate of {rate} and number of epochs of {num_epochs}')
        print("\n")

