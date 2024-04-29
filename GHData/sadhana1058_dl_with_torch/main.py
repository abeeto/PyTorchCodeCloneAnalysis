from tqdm import tqdm
import torch

def train(model, device, train_loader, optimizer, scheduler,criterion, epoch, train_losses, train_accuracy):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss = 0
  
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)
    
    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_loss += loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()
	
    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_loss /= len(train_loader.dataset)    
  train_losses.append(train_loss)
  train_accuracy.append(100. * correct / len(train_loader.dataset)) 

def test(model, device, criterion, test_loader,test_losses, test_accuracy ):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracy.append(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def start_training_the_model(model, device, trainloader, testloader,criterion,optimizer, train_losses, train_accuracy, test_losses, test_accuracy,EPOCHS):
	for epoch in range(EPOCHS):
		print("EPOCH:", epoch+1, 'LR:',optimizer.param_groups[0]['lr'])#printing lr for each epoch
		train(model, device, trainloader, optimizer, criterion, epoch,train_losses,train_accuracy )
		test(model, device, criterion, testloader,test_losses, test_accuracy )
