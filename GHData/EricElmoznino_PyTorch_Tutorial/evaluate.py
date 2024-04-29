import torch
from torch.utils.data import DataLoader
import os

from models import Model, PretrainedModel
from MNISTDataset import MNISTDataset
import utils


def classification_accuracy(predictions, targets):
    _, top_classes = predictions.max(1)     # Returns (value, index)
    n_correct = (top_classes == targets).sum().item()
    accuracy = n_correct / len(targets)
    return accuracy


def evaluate_dataset(model, dataloader):
    model.eval()    # Certain layers (e.g. BatchNorm and Dropout) behave differently

    running_losses = None
    with torch.no_grad():   # Do not make dynamic graph, since not backpropagating
        for data in dataloader:

            # Obtain the inputs/targets
            images, labels = data
            labels = labels.view(
                -1)  # Labels must be 1 dimensional with class indices, but DataLoader creates batch dimension
            images, labels = utils.device([images, labels])

            predictions = model(images)

            # Compute error metrics and update running losses
            accuracy = classification_accuracy(predictions, labels)
            running_losses = utils.update_losses(running_losses, {'accuracy': accuracy}, len(dataloader))

    model.train()
    return running_losses


if __name__ == '__main__':
    use_custom_model = True

    if use_custom_model:
        model = Model(in_channels=3, n_classes=10)
        model_name = 'custom_model'
    else:
        model = PretrainedModel(n_classes=10)
        model_name = 'pretrained_model'
    if torch.cuda.is_available():
        model.cuda()

    utils.load_model(model, os.path.join('saved_runs', model_name))

    test_set = MNISTDataset('data/test', resolution=[32, 32])
    test_loader = DataLoader(test_set, num_workers=2)

    print('Evaluation')
    eval_losses = evaluate_dataset(model, test_loader)
    print(eval_losses)
