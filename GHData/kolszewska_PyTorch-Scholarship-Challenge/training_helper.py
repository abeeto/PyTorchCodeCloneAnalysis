import logging

import torch
from tensorboardX import SummaryWriter

writer = SummaryWriter()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train(train_loader, validation_loader, num_epochs, total_training_batches, model, criterion, optimizer, scheduler):
    batch_number = 0
    step_number = 0

    for epoch in range(num_epochs):
        train_running_loss = 0
        train_accuracy = 0
        scheduler.step()
        for images, labels in train_loader:
            if batch_number % 10 == 0:
                logging.info('Batch number {}/{}...'.format(batch_number, total_training_batches))
            batch_number += 1
            step_number += 1
            # Pass this computations to selected device
            images = images.cuda()
            labels = labels.cuda()

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # Forwards pass, then backward pass, then update weights
            probabilities = model.forward(images)
            loss = criterion(probabilities, labels)
            loss.backward()
            optimizer.step()

            # Get the class probabilities
            ps = torch.nn.functional.softmax(probabilities, dim=1)

            # Get top probabilities
            top_probability, top_class = ps.topk(1, dim=1)

            # Comparing one element in each row of top_class with
            # each of the labels, and return True/False
            equals = top_class == labels.view(*top_class.shape)

            # Number of correct predictions
            train_accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

            train_running_loss += loss.item()
        else:
            validation_running_loss = 0
            validation_accuracy = 0
            # Turn off gradients for testing
            with torch.no_grad():
                # set model to evaluation mode
                model.eval()
                for images, labels in validation_loader:
                    # Pass this computations to selected device
                    images = images.cuda()
                    labels = labels.cuda()

                    probabilities = model.forward(images)
                    validation_running_loss += criterion(probabilities, labels)

                    # Get the class probabilities
                    ps = torch.nn.functional.softmax(probabilities, dim=1)

                    # Get top probabilities
                    top_probability, top_class = ps.topk(1, dim=1)

                    # Comparing one element in each row of top_class with
                    # each of the labels, and return True/False
                    equals = top_class == labels.view(*top_class.shape)

                    # Number of correct predictions
                    validation_accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

            # Set model to train mode
            model.train()

            # Calculating accuracy
            validation_accuracy = (validation_accuracy / validation_loader.sampler.num_samples * 100)
            train_accuracy = (train_accuracy / train_loader.batch_sampler.sampler.num_samples * 100)

            # Saving losses and accuracy
            writer.add_scalar('data/train_loss', train_running_loss, step_number)
            writer.add_scalar('data/train_accuracy', train_accuracy, step_number)
            writer.add_scalar('data/validation_loss', validation_running_loss, step_number)
            writer.add_scalar('data/validation_accuracy', validation_accuracy, step_number)

            logging.info("Epoch: {}/{}.. ".format(epoch + 1, num_epochs))
            logging.info("Training Loss: {:.3f}.. ".format(train_running_loss))
            logging.info("Training Accuracy: {:.3f}%".format(train_accuracy))
            logging.info("Validation Loss: {:.3f}.. ".format(validation_running_loss))
            logging.info("Validation Accuracy: {:.3f}%".format(validation_accuracy))

            batch_number = 0
