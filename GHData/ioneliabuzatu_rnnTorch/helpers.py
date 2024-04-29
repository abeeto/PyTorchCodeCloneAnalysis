import prepareData
import torch.nn as nn
import torch
import helpers
import random

# outcome from data preparation
all_letters, n_letters, category_lines, all_categories = prepareData.PrepareData().outcome()
n_categories = len(all_categories)


# turning data into 2D tensors to make any use of them
def letterToIndex(letter):
    """
    :returns letter index from all_letters, e.g. "a" = 0
    """
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor



# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for letter_idx, letter in enumerate(line):
        tensor[letter_idx][0][letterToIndex(letter)] = 1
    return tensor


# print(lineToTensor("Manuel")[0][0][0])


# Tensor.topk to get the index of the greatest value
def categoryTopIndex(output):
    topNumber, topIndex = output.topk(1)
    categoryIndex = topIndex[0].item()
    prediction = all_categories[categoryIndex]
    return prediction, categoryIndex

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    # print('category =', category_tensor, '/ line =', line_tensor)


# if __name__ == '__main__':
