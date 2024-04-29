import torch
from tqdm import tqdm
from easydict import EasyDict
import os

def train_model(train_config):
    '''
    train_config = EasyDict(
        device      = computing device
        num_epochs  = maximum number of epochs
        batch_size  = training batch size
        num_workers = number of data threads
        model_path  = output model path
        datasets    = EasyDict(train, test) training and testing sets
        model       = model to train, moved to computing device
        criterion   = callable loss function 
        optimizer   = specified optimizer
        scheduler   = specified scheduler
        additional_preprocess = func(images, labels) -> (images, labels)
    )
    return:
      save last epoch model and best model to train_config.model_path
      train_config.model is the trained model
    '''
    def get_progress_bar(loader):
        '''
        loader: a DataLoader
        return: a tqdm progress bar with total steps equal len(loader)
        '''
        return tqdm(enumerate(loader), total=len(loader))

    def optimizer_step(loss):
        '''
        carry out a training step using optimizer 
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def to_device(images, labels):
        '''
        return: tuple(image, labels) that are moved to computing device
        '''
        images = images.to(train_config.device)
        labels = labels.to(train_config.device)
        return images, labels

    def train_one_epoch(epoch):
        '''
        carry out a training epoch with progress bar
        '''
        model.train()
        pbar = get_progress_bar(loaders.train)
        for i, (images, labels) in pbar:
            if 'additional_preprocess' in train_config:
                images, labels = train_config.additional_preprocess(images, labels)
            images, labels = to_device(images, labels)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            optimizer_step(loss)
            scheduler.step()

            pbar.set_description(f'train epoch {epoch} loss {loss.item():.4f}')

    def init_metric():
        '''
        return: a zero dict(total, correct) to start accumulating
        '''
        return EasyDict(dict(total=0, correct=0))
    
    def accumulate_metric(predicted, labels, metric):
        '''
        update a metric = dict(total, correct) with test results from (predicted, labels)
        '''
        metric.total += labels.size(0)
        metric.correct += (predicted == labels).sum().item()
    
    def test_one_epoch(epoch):
        '''
        carry out a testing epoch with progress bar
        accumulating accuracy in all batches
        '''
        model.eval()
        pbar = get_progress_bar(loaders.test)
        metric = init_metric()           
        with torch.no_grad():
            for i, (images, labels) in pbar:
                if 'additional_preprocess' in train_config:
                    images, labels = train_config.additional_preprocess(images, labels)
                images, labels = to_device(images, labels)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                accumulate_metric(predicted, labels, metric)
                
                pbar.set_description(f'test  epoch {epoch} accuracy {100*metric.correct/metric.total:.2f}%')
            save_model(accuracy=metric.correct/metric.total, epoch=epoch)

    def save_model(accuracy, epoch):
        '''
        save current model and save current best model
        '''
        nonlocal current_best
        data_to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': accuracy,
        }
        torch.save(data_to_save, f'{train_config.model_path}.ckpt')
        if accuracy > current_best:
            last_file = f'{train_config.model_path}_best_{current_best:.4f}.ckpt'
            if os.path.exists(last_file):
                os.remove(last_file)
            print(f'val_acc improved from {current_best*100:.2f}% to {accuracy*100:.2f}%')
            torch.save(data_to_save, f'{train_config.model_path}_best_{accuracy:.4f}.ckpt')
            current_best = accuracy

    # training main code
    model = train_config.model
    optimizer = train_config.optimizer
    scheduler = train_config.scheduler
    criterion = train_config.criterion
    datasets = train_config.datasets
    
    # create data loaders
    loaders = EasyDict(dict(
        train=torch.utils.data.DataLoader(datasets.train, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers),
        test=torch.utils.data.DataLoader(datasets.test, batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers),
    ))
    
    # start training
    current_best = float('-inf')
    for epoch in range(train_config.num_epochs):
        train_one_epoch(epoch)
        test_one_epoch(epoch)
