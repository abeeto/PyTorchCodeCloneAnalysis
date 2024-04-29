import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
from torchtext.vocab import Vectors
from model import CNN

# Seed for random.
seed = int(time.time())
# Choose training device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(word_vec_path, cache_path: str, max_vocab_size: int = 25000):
    """Load data from local files.
    When executed first time without '.data/' folder in path,
    the text data will be automatically downloaded.

    :param word_vec_path: str, path of pre-trained word vector data file
    :param max_vocab_size: int, maximum vocabulary size
    :param cache_path: str, path of word vector cache storage
    :return: text, label and tuple of (train, valid, test) data
    """

    torch.manual_seed(seed)
    # Improve efficiency by using deterministic convolution.
    torch.backends.cudnn.deterministic = True
    # Improve efficiency as the input size of model doesn't change.
    torch.backends.cudnn.benchmark = True
    text = data.Field(tokenize='spacy')
    label = data.LabelField(dtype=torch.float)
    # Split dataset into train, valid and test sets.
    train_data, test_data = datasets.IMDB.splits(text, label)
    # Load word vector from local file.
    word_vec = Vectors(name=word_vec_path, cache=cache_path)
    # Build vocabulary.
    text.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors=word_vec,
                     unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data)
    return text, label, (train_data, test_data)


def get_iterator(dataset, batch_size: int = 64):
    """Get iterators of dataset tuple.

    :param dataset: tuple, (train, valid) data
    :param batch_size: int, size of each batch
    :return: tuple, iterators of the dataset tuple
    """

    # Split train data and validation data.
    return data.Iterator(dataset, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False)


def train(model: CNN, train_set, optimizer, criterion: nn.modules.loss, model_path: str, epochs: int = 10):
    """Train epochs using train data and validation data.

    :param model: CNN model
    :param train_set: train data
    :param optimizer: optimizer of model, usually adam
    :param criterion: loss function of model
    :param epochs: int, times of training on the total set, default is 5
    :param model_path: str, path of model storage
    :return: None
    """

    # Record the minimum loss value, start from infinity.
    best_valid_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_data, valid_data = train_set.split(random_state=random.seed(seed))
        train_loss, train_acc = train_epoch(model, get_iterator(train_data), optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, get_iterator(valid_data), criterion)
        # Save the best model for further training.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
        print('\nEpoch: {} | Epoch Time: {:.2f} s'.format(epoch, time.time() - start_time))
        print('\tTrain: loss = {:.2f}, accuracy = {:.2f}%'.format(train_loss, train_acc * 100))
        print('\tValid: loss = {:.2f}, accuracy = {:.2f}%'.format(valid_loss, valid_acc * 100))


def calc_accuracy(pred: torch.Tensor, y: torch.Tensor):
    """Calculate accuracy of each batch.

    :param pred: Tensor, output of model on test input
    :param y: true tags, use 0 / 1 uni-variable to represent class
    :return: float value of accuracy between [0, 1]
    """

    rounded_preds = torch.round(torch.sigmoid(pred))
    correct = (rounded_preds == y).float()
    accuracy = correct.sum() / y.shape[0]
    return accuracy


def train_epoch(model, iterator, optimizer, criterion):
    """Train each epoch, for both train set and valid set.

    :param model: CNN model
    :param iterator: iterator of training data
    :param optimizer: optimizer of model, usually adam
    :param criterion: loss function of model
    :return: float value of total loss, and accuracy of this epoch
    """

    epoch_loss, epoch_acc = 0, 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = calc_accuracy(predictions, batch.label)
        # Backward propagation.
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """Make predictions and check its loss / accuracy on test set.

    :param model: CNN model
    :param iterator: iterator of test data
    :param criterion: loss function of model
    :return: float value of total loss, and accuracy of predictions
    """

    epoch_loss, epoch_acc = 0, 0
    # Change mode to eval.
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = calc_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    # Dimension of word vector to use, choose from [50, 100, 200, 300].
    embedding_dim = 200
    word_vec_path = './glove.6B/glove.6B.{}d.txt'.format(embedding_dim)
    model_path = './saved_models/model_{}d.pt'.format(embedding_dim)
    cache_path = './vec_cache/'

    # Load and init data.
    text, label, dataset = load_data(word_vec_path, cache_path)

    # Build model with configurations.
    input_dim = len(text.vocab)
    n_filters = 100
    filter_sizes = [3, 4, 5]
    output_dim = 1
    dropout = 0.5
    pad_idx = text.vocab.stoi[text.pad_token]
    model = CNN(input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)

    # Copy pre-trained word vectors
    model.embedding.weight.data.copy_(text.vocab.vectors)
    # Deal with unknown word & padding words: 0 vectors.
    unk_idx = text.vocab.stoi[text.unk_token]
    model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

    # Use adam to optimize training.
    optimizer = optim.Adam(model.parameters())
    # Binary Cross Entropy using logistic loss.
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # Try to load model from previous training.
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # Train model on train data.
    train(model, dataset[0], optimizer, criterion, model_path, 5)

    # Load the best model and check accuracy on test data.
    model.load_state_dict(torch.load(model_path))
    test_it = get_iterator(dataset[1])
    test_loss, test_acc = evaluate(model, test_it, criterion)
    print('\nTest Loss: {:.2f} | Test Accuracy: {:.2f}%'.format(test_loss, test_acc * 100))


if __name__ == '__main__':
    main()
