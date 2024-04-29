import argparse
from tqdm import tqdm, trange
import logging as logging

import torch 
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchsummary import summary

from utils import load_dataset
from model.classifier import Classifier

# -------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

# -------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cpu")
    return parser.parse_args()

# -------------------------------------------------------------------
def save_model(model, path="weights/vit.pt"):
    torch.save(model.state_dict(), path)


def train(opt, model, train_loader, test_loader):
    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = CrossEntropyLoss()

    for epoch in trange(opt.epochs, desc = "Training"):
        train_loss = 0.0
        correct, total = 0, 0 
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x = x.to(opt.device)
            y = y.to(opt.device)

            y_hat = model(x)
            loss = criterion(y_hat, y) / len(x)
            train_loss += loss.item()

            total += len(x)
            correct += torch.sum((torch.argmax(y_hat.data, dim=1) == y)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        # print(f'\nEpoch: {epoch+1}/{opt.epochs}, Train Loss: {train_loss}, Train Accuracy: {correct/total*100:.2f}%')
        logging.info(f'\nEpoch: {epoch+1}/{opt.epochs}, Train Loss: {train_loss}, Train Accuracy: {correct/total*100:.2f}%')
        test_correct, test_total = 0, 0 
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing', leave=False):
                x, y = batch
                x = x.to(opt.device)
                y = y.to(opt.device)

                y_hat = model(x)
                loss = criterion(y_hat, y) / len(x)
                test_loss += loss.item()

                test_total += len(x)
                test_correct += torch.sum((torch.argmax(y_hat.data, dim=1) == y)).item()

        # print(f'\nTest Loss: {test_loss}, Accuracy: {test_correct/test_total*100:.2f}% \
        #     \n\n--------------------------------------------------------------------')
        
        logging.info(f'\nTest Loss: {test_loss}, Accuracy: {test_correct/test_total*100:.2f}% \
            \n\n--------------------------------------------------------------------')
        
    model_name = f"weights/cnn.pt"
    save_model(model, model_name)


if __name__ == '__main__':
    # Get arguments
    opt = get_args()

    # Load MNIST dataset into DataLoader
    train_load, test_load = load_dataset(opt.batch_size) 
    logging.info("Dataset loaded\n")

    # Load model
    model = Classifier(
        input_dim=(1, 28, 28),
        num_classes=opt.num_classes
    )
    logging.info("Model loaded\n")
    summary(model, (1, 28, 28))

    # Train model
    logging.info("Training model\n")
    train(opt, model, train_load, test_load)