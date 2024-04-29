import torch
from model import RNN
from data import lineToTensor, all_categories, n_letters, n_categories

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
rnn.load_state_dict(torch.load('char-rnn-classification.pth'))
rnn.eval()

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(line, n_predictions=3):
    output = evaluate(lineToTensor(line))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    topv = torch.exp(topv)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value.item(), all_categories[category_index]])

    return predictions


def main(input_dict):
    assert isinstance(input_dict, dict), 'Input must be a dictionary'
    line = input_dict.get('name', '')
    n_predictions = input_dict.get('n_predictions', 1)
    return predict(line, n_predictions)


# print(main(dict(name='Miyazaki', n_predictions=3)))

