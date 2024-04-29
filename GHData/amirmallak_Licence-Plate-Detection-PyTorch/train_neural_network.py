import time
import torch
import numpy as np

from torch import Tensor
# from intersection_over_union import iou_metric


def train(model, train_loader, train_dataset, val_loader, val_dataset, train_step, loss_fn, width: int) -> tuple:
    n_epochs = 50
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    accuracy_percentage = 10e-2  # The radius of accuracy circle is 10% of image dimensions
    # tolerance = 92e-2
    # print(model.state_dict())

    times = []

    print('\nTraining the Neural Network Model...\n')
    start_time = time.time()
    for epoch in range(n_epochs):
        start_epoch_time = time.time()

        batch_losses = []
        correct = 0
        for x_batch, y_batch in train_loader:
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.to(device)

            output: Tensor = model(x_batch)  # model.forward(x_batch)
            loss = train_step(x_batch, y_batch)
            batch_losses.append(loss)

            # Check if the mean radius Euclidean metric is bounded below the accuracy percentage specified
            difference_bbox: np.ndarray = np.abs(((output - y_batch) * width).detach().numpy())
            correct += ((np.sum(difference_bbox, axis=1) / 4) < (accuracy_percentage * width)).sum()

            # correct = iou_metric(y_batch, output).detach().numpy()
            # correct += np.sum(np.diagonal(correct_2_array) > tolerance)

        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        accuracy = 100 * correct / len(train_dataset)
        training_accuracies.append(accuracy)

        with torch.no_grad():
            val_losses = []
            correct = 0
            for x_val, y_val in val_loader:
                # x_val = x_val.to(device)
                # y_val = y_val.to(device)
                # output: Tensor = model(x_val)  # model.forward(x_batch)

                model.eval()
                yhat: Tensor = model(x_val)
                val_loss = loss_fn(model, y_val, yhat)  # .item()
                val_losses.append(val_loss)

                # Check if the mean radius Euclidean metric is bounded below the accuracy percentage specified
                difference_bbox: np.ndarray = np.abs(((yhat - y_val) * width).detach().numpy())
                correct += ((np.sum(difference_bbox, axis=1) / 4) < (accuracy_percentage * width)).sum()

            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)
            accuracy = 100 * correct / len(val_dataset)
            validation_accuracies.append(accuracy)

        print(f"[{epoch + 1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
        print(f"[{epoch + 1}] Training accuracy: {training_accuracies[-1]:.3f}\t "
              f"Validation accuracy: {validation_accuracies[-1]:.3f}")

        end_epoch_time = time.time()
        elapsed_epoch_time = end_epoch_time - start_epoch_time
        times.append(elapsed_epoch_time)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # print(model.state_dict())

    print('\nFinished Training!\n')

    print(f'Training the Neural Network Model Time: {elapsed_time / 60 :.2f} [min]')
    print(f'The average time for each epoch: {(sum(times) / n_epochs) / 60 :.2f} [min]')

    return training_accuracies, training_losses, validation_accuracies, validation_losses
