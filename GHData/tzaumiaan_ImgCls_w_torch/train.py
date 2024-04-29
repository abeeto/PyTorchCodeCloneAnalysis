from data_utils import load_data

from datetime import datetime
import torch

def apply_cuda(obj):
  if torch.cuda.is_available():
    obj = obj.cuda()
  return obj

def apply_var(obj):
  return torch.autograd.Variable(obj)

def train(batch_size=50, lr=0.01, data_folder='data', dataset_name='mnist', model_name='lenet', max_epochs=10, log_freq=100):
  # model definition
  if model_name == 'lenet':
    from model.lenet import LeNet
    model = LeNet()
  else:
    from model.modelzoo import create_model
    model, input_size = create_model(model_name, n_classes=120)
  model = apply_cuda(model)
  
  # data source
  if dataset_name == 'mnist':
    train_loader = load_data('train', batch_size, data_folder, dataset_name)
    eval_loader = load_data('test', batch_size, data_folder, dataset_name)
  else:
    train_loader = load_data('train', batch_size, data_folder, dataset_name, input_size)
    eval_loader = load_data('test', batch_size, data_folder, dataset_name, input_size)
  n_batches_train = len(train_loader)
  n_batches_eval = len(eval_loader)
  print(
      datetime.now(),
      'batch size = {}'.format(batch_size),
      'number of batches for training = {}'.format(n_batches_train),
      'number of batches for evaluation = {}'.format(n_batches_eval))

  # optimizer and loss definition
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  
  for epoch in range(max_epochs):
    print(datetime.now(), 'epoch: {}/{}'.format(epoch+1, max_epochs))
    
    # training set
    print('==== training phase ====')
    avg_loss = float(0)
    avg_acc = float(0)
    model.train()
    for step, (images, labels) in enumerate(train_loader):
      optimizer.zero_grad()
      images, labels = apply_cuda(images), apply_cuda(labels)
      images, labels = apply_var(images), apply_var(labels)
      # forward pass
      if model_name == 'inception_v3':
        logits, aux_logits = model(images)
        loss1 = criterion(logits, labels)
        loss2 = criterion(aux_logits, labels)
        loss = loss1 + 0.4*loss2
      else:
        logits = model(images)
        loss = criterion(logits, labels)
      _, pred = torch.max(logits.data, 1)
      bs_ = labels.data.size()[0]
      match_count = (pred == labels.data).sum()
      accuracy = float(match_count)/float(bs_)
      avg_loss += loss.item()/float(n_batches_train)
      avg_acc += accuracy/float(n_batches_train)
      # backward pass
      loss.backward()
      optimizer.step()
      if (step+1) % log_freq == 0:
        print(
            datetime.now(),
            'training step: {}/{}'.format(step+1, n_batches_train),
            'loss={:.5f}'.format(loss.item()),
            'acc={:.4f}'.format(accuracy))
    print(
        datetime.now(),
        'training ends with avg loss={:.5f}'.format(avg_loss),
        'and avg acc={:.4f}'.format(avg_acc))
    # validation set
    print('==== validation phase ====')
    avg_acc = float(0)
    model.eval()
    for images, labels in eval_loader:
      images, labels = apply_cuda(images), apply_cuda(labels)
      logits = model(images)
      _, pred = torch.max(logits.data, 1)
      bs_ = labels.data.size()[0]
      match_count = (pred == labels.data).sum()
      accuracy = float(match_count)/float(bs_)
      avg_acc += accuracy/float(n_batches_eval)
    print(
        datetime.now(),
        'evaluation results: acc={:.4f}'.format(avg_acc))
    
    # save the model for every epoch
    ckpt_path = '{}_{}_bs{}_lr{}_ep{}.pth'.format(
        model_name,
        dataset_name,
        batch_size,
        lr,
        epoch)
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'avg_loss': avg_loss,
        'avg_acc': avg_acc}, ckpt_path)

if __name__ == '__main__':
  from config import lr, bs, model_name
  train(
      batch_size=bs,
      lr=lr,
      data_folder='data',
      dataset_name='sdd',
      model_name=model_name,
      max_epochs=1,
      log_freq=100)

