import torch
from tqdm import tqdm
from utils import seed_everything, get_device, AverageMeter
from config import TrainConfig
from model import MNISTClassifier


class MNISTTrainer:
    def __init__(
        self, 
        config: TrainConfig, 
        model: MNISTClassifier, 
        train_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: torch.utils.data.DataLoader, 
        criterion: torch.nn.modules.loss,
        optimizer: torch.optim, 
    ):
        self.device = get_device()
        self.config = config
        self.model = model.to(self.device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        seed_everything(self.config.seed)
        print(f'Trainer init complete!')


    def _train_one_epoch(self):
        train_acc, train_loss = AverageMeter(), AverageMeter()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, (img, label) in loop:
            img = img.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(img)

            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, dim=-1)
            acc = torch.sum(preds == label).item() / len(label)

            train_acc.update(acc)
            train_loss.update(loss, len(label))
        
        return train_acc.avg, train_loss.avg
    
    
    def _print_train_log(self, epoch, train_acc, train_loss, val_acc, val_loss):
        print(f'Epoch {epoch+1} | Train Accuracy : {train_acc:.4f}, Train Loss : {train_loss:.4f} | Val Accuracy : {val_acc:.4f}, Val Loss : {val_loss:}')


    def train(self):
        print(f'Training Start...')
        self.model.train()
        for epoch in range(self.config.epochs):
            train_acc, train_loss = self._train_one_epoch()
            val_acc, val_loss = self.validate()
            self._print_train_log(epoch, train_acc, train_loss, val_acc, val_loss)


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_acc, val_loss = AverageMeter(), AverageMeter()
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for i, (img, label) in loop:
            img = img.to(self.device)
            label = label.to(self.device)

            outputs = self.model(img)
            loss = self.criterion(outputs, label)

            preds = torch.argmax(outputs, dim=-1)
            acc = torch.sum(preds == label).item() / len(label)

            val_acc.update(acc)
            val_loss.update(loss, len(label))

        return val_acc.avg, val_loss.avg


    def save_model(self):
        print(f'Saving model...')
        torch.save(self.model.state_dict(), self.config.model_save_path)
        print('Saving model done!')
