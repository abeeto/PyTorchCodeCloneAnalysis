"""Helper functions to convert data between various types/representations"""

import torch
import string
from hyperparameters import hps

# create dictionaries to convert between characters and ints
chars = list(string.ascii_lowercase+'_') # underscore represents end of name
ints = range(27)
char_to_i = {char: i for char, i in zip(chars, ints)}
i_to_char = {i: char for char, i in char_to_i.items()}

def validate_letter_input(letter):
    """Make sure the input letter is actually a letter

    * letter: the letter to validate
    """

    # verify the input is of type string
    if type(letter) is not str:
        raise Exception('{} is not a string'.format(letter))
    # verify the input is one of the characters that the RNN can process
    if letter not in char_to_i.keys():
        raise Exception('{} is not a lowercase letter or the underscore'.format(letter))

def letter_to_onehot(letter):
    """Convert a letter to a onehot tensor

    * letter: the letter to convert
    """

    validate_letter_input(letter)
    # create a tensor with a 1 at the letter's index and zeros everywhere else
    i = char_to_i[letter]
    onehot = torch.zeros(hps['onehot_length'])
    onehot[i] = 1
    # reshape into the shape that the LSTM module requires for inputs
    onehot = onehot.view(1, -1)

    return onehot

def letter_to_category(letter):
    """Convert a letter to a classification category

    * letter: the letter to convert
    """

    validate_letter_input(letter)
    # create the tensor in the shape that CrossEntropyLoss requires
    category = torch.tensor([char_to_i[letter]])

    return category

def category_to_letter(category):
    """Convert a classification category to a letter

    * category: the category to convert
    """

    # verify the input is of type int
    if type(category) is not int:
        raise Exception('{} is not an int'.format(category))
    # verify the input is in the range (0, 26)
    if category not in range(27):
        raise Exception('{} is not in the range (0, 26)'.format(category))

    # simply use the dictionary
    letter = i_to_char[category]

    return letter

def language_to_category(language):
    ltc = {'english': 0, 'french': 1, 'spanish': 2}
    category = ltc[language]
    # create the tensor in the shape that CrossEntropyLoss requires
    category = torch.tensor([category])
    return category

def category_to_language(category):
    ctl = {0: 'english', 1: 'french', 2: 'spanish'}
    language = ctl[category]
    return language

def pair_to_xy(pair):
    word, language = pair

    # create lists to be iterated through during training step
    # and convert to the correct format for inputs/outputs
    xs = [letter_to_onehot(letter) for letter in word+'_']
    y = language_to_category(language)

    return xs, y
