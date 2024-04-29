import torch
import torch.nn as nn
from torch.nn.functional import softmax
import helpers
import random
from hyperparameters import hps
import data

model = nn.LSTM(hps['onehot_length'], 3, hps['lstm_layers'])
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

dataset = data.get_pairs('datasets/test_set.csv')

correct_preds = []
wrong_preds = []

with torch.no_grad():
    n_correct = 0
    n_wrong = 0

    # run through the whole test set once
    for pair in dataset:
        xs, y = helpers.pair_to_xy(pair)
        # make a tensor for the whole sequence
        xs = torch.stack(xs)
        y_pred, _ = model(xs)
        y_pred = y_pred[-1].view(-1)
        sm = softmax(y_pred, dim=0)
        pred_cat = torch.argmax(sm).item()
        pred_language = helpers.category_to_language(pred_cat)

        word = pair[0]
        target_language = pair[1]
        # print('Word: {}'.format(word))
        # print('Target language: {}'.format(target_language))
        # print('Predicted language: {}'.format(pred_language))

        if pred_language == target_language:
            correct_preds.append((word, target_language))
            n_correct += 1
        else:
            wrong_preds.append((word, target_language, pred_language))
            n_wrong += 1

    print('Correct: {}'.format(n_correct))
    print('Wrong: {}'.format(n_wrong))

    accuracy = n_correct/len(dataset)
    print('Accuracy: {:.2}'.format(accuracy))
    # for word, target_language, pred_language in wrong_preds:
    #     print('{}: {}, not {}'.format(word, target_language, pred_language))
