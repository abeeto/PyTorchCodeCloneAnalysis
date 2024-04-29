import io

import spacy
from spacy.symbols import ORTH
from torchtext import data
from tqdm import tqdm

from Field import Example


class LanguageModelingDataset(data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self,
                 path,
                 text_field,
                 encoding='utf-8',
                 sequence_len=15,
                 **kwargs):
        """Create a LanguageModelingDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            fields: Dictionary containing keyword
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        fields = [('text', text_field)]
        text = []

        with io.open(path, encoding=encoding) as f:

            for line in f:
                temp = []
                temp += ['<bos>']  # add begining of the sentence token
                temp += text_field.preprocess(line)
                temp.append(u'<eos>')
                if len(temp) > sequence_len:
                    temp = temp[:sequence_len - 1]
                    temp.append(u'<eos>')  # add end of the sentence token
                if len(temp) < sequence_len:
                    temp += ['<pad>'] * (sequence_len - len(temp))  # already pad everything
                text += temp
        examples = [Example.fromlist([text], fields)]

        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)


def text_cleaner(file, out_file):
    with io.open(file, encoding='utf-8') as f:
        with open(out_file, 'w') as fout:
            my_tok = spacy.load('en_core_web_sm')
            for i, line in enumerate(tqdm(f)):
                if len(line) > 5:
                    if line[0] == "=" or line[1] == "=":
                        continue
                    doc = my_tok(line)
                    for sent in doc.sents:
                        fout.write(str(sent).replace("\n", "") + "\n")


def create_tokenizer():
    '''
    just a custom tokenizer for a simple corpus
    :return: spacy tokenizer
    '''
    my_tok = spacy.load('en_core_web_sm')
    my_tok.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
    my_tok.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
    my_tok.tokenizer.add_special_case('< unk >', [{ORTH: '<unk>'}])
    my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
    my_tok.tokenizer.add_special_case('<bow>', [{ORTH: '<bow>'}])
    my_tok.tokenizer.add_special_case('<eow>', [{ORTH: '<eow>'}])

    def spacy_tok(x):
        return [tok.text.lower() for tok in my_tok.tokenizer(x)]

    return spacy_tok


def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    ids = id_tensor.view(-1)
    batch = [vocab.itos[ind] for ind in ids]  # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)
