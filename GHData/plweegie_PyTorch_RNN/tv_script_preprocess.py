import helper
import numpy as np
from collections import Counter


data_dir = 'data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)

view_line_range = (0, 10)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    int_to_vocab = {ii: word for word, ii in vocab_to_int.items()}
    
    # return tuple
    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
  
    dict = {'.': "||Period||",
            ',': "||Comma||",
            '"': "||Quotation_Mark",
            ';': "||Semicolon||",
            '!': "||Exclamation_Mark||",
            '?': "||Question_Mark||",
            '(': "||Left_Parentheses||",
            ')': "||Right_Parentheses||",
            '-': "||Dash||",
            '\n': "||Return||"}
    
    return dict


helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
