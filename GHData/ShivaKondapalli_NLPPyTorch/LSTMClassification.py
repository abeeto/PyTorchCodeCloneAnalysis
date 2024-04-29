from LSTMNetwork import LSTM
from RNNClassification import get_files, readfiles,  n_letters, nametotensor, time_taken
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import torch
torch.manual_seed(7)

path = "data/Gender/"
ext = ".txt"

sex_to_name = dict()
all_categories = []

for f in get_files(path):
    category = f.split('/')[2].split('.')[0]
    all_categories.append(category)
    lines = readfiles(f)
    sex_to_name[category] = lines

n_categories = len(all_categories)

n_hidden = 256
lstm = LSTM(n_letters, n_hidden, n_categories)


def genderfromoutput(output):
    top_v, top_i = output.topk(1)
    cat_i = top_i[0].item()
    return all_categories[cat_i], cat_i


def randomtrainningexample():
    sex = np.random.choice(all_categories)
    name = np.random.choice(sex_to_name[sex])
    sex_tensor = torch.tensor([all_categories.index(sex)], dtype=torch.long)
    name_tensor = nametotensor(name)
    return sex, name, sex_tensor, name_tensor


learning_rate = 0.007
criterion_lstm = nn.CrossEntropyLoss()


def train_lstm(sex_tensor, name_tensor):
    lstm.zero_grad()

    output = lstm.forward(name_tensor)

    loss = criterion_lstm(output.squeeze(1), sex_tensor)
    loss.backward()

    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)  # can also use torch.optim() if you so choose to

    return output, loss.item()


def evaluate(name_tensor, model):

    out = model.forward(name_tensor)

    return out


def predict(name, model, n_predictions=3):

    with torch.no_grad():
        output = evaluate(nametotensor(name), model)

        output = output.squeeze(1)

        top_n, top_i = output.topk(n_predictions, 1, True)
        predictions_lst = []

        for i in range(n_predictions):
            val = top_n[0][i]
            cat_idx = top_i[0][i].item()
            print(f'Value: {val.item()}, language: {all_categories[cat_idx]}')
            predictions_lst.append([val, all_categories[cat_idx]])


def main():

    for i in range(10):
        sex, name, sex_tensor, name_tensor = randomtrainningexample()
        print(f'sex: {sex}, name: {name}')

    print(f'lstm: {lstm}')

    # Training
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    current_loss = 0
    all_losses = []

    start = time.time()

    for i in range(1, n_iters+1):
        sex, name, sex_tensor, name_tensor = randomtrainningexample()
        output, loss = train_lstm(sex_tensor, name_tensor)
        current_loss += loss

        if n_iters % print_every == 0:
            pred, pred_i = genderfromoutput(output)
            prediction = 'True' if pred == sex else f'False, correct one is {sex}'
            print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, time_taken(start), loss, name, pred, prediction))

        if i % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0

    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Add one to each row: the real category and each column: the predicted category.
    # The darker the principal diagonal, the better the model.
    for i in range(n_confusion):
        sex, name, sex_tensor, name_tensor = randomtrainningexample()
        output = evaluate(name_tensor, lstm)
        guess, guess_i = genderfromoutput(output)
        real_category_i = all_categories.index(category)
        confusion[real_category_i][guess_i] += 1

    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up fig, axes.
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title('Confusion Matrix for two classes')
    cax = ax1.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set the labels for x and y axes
    ax1.set_xticklabels([''] + all_categories, rotation=90)
    ax1.set_yticklabels([''] + all_categories)

    # Major tick locations on the axis are set.
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Plot Vanilla Rnn losses
    ax1 = fig.add_subplot(122)
    ax1.set_title('LSTM Losses')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Losses')
    ax1.plot(all_losses)
    plt.show()


if __name__ == "__main__":
    main()


