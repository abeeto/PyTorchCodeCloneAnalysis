
import torch
from tqdm import tqdm
from typing import Tuple
import torch.nn.functional as functional


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    train_loss, train_acc = 0, 0

    for i, (images, labels) in enumerate(dataloader):
        # Move tensors to the configured device
        images, labels = images.to(device), labels.to(device)
        one_hot_labels = functional.one_hot(labels, num_classes=100).float()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, one_hot_labels)
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()/len(labels)

    # Calculate average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return (train_loss, train_acc)


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            one_hot_labels = functional.one_hot(
                labels, num_classes=100).float()
            outputs = model(images)

            # Calculate loss
            loss = loss_fn(outputs, one_hot_labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_acc += (predicted == labels).sum().item()/len(labels)
            del images, labels, outputs

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return (test_loss, test_acc)


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
		  writer: torch.utils.tensorboard.writer.SummaryWriter,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):

	for epoch in tqdm(range(epochs)):
		train_loss, train_acc = train_step(model, train_loader,
											loss_fn, optimizer, device)
		
		# Write train loss and accuracy into tensorboard
		writer.add_scalars("train/loss+acc", {'loss': train_loss,
											'acc': train_acc}, epoch)
		writer.flush()

		print('Epoch [{}/{}], Loss: {:.4f}'
				.format(epoch+1, epochs, train_loss))

		test_loss, test_acc = test_step(model, valid_loader, loss_fn, device)
		
		# Write validation loss and accuracy into tensorboard
		writer.add_scalars("test/loss+acc", {'loss': test_loss,
											'acc': test_acc}, epoch)
		writer.flush()

	