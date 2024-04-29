from torchtext import data
from torchtext.vocab import GloVe, FastText

from datasets import WikiSyntheticGeneral


def load_dataset(dataset_name='WikiSyntheticGeneral', splits=None, tokenize_func='split',
                 embedding_func='glove', batch_size=32):
    """
    Arguments:
        dataset_name: which dataset to use, for training (and testing) use WikiSyntheticGeneral,
            then with trained models try out WikiSyntheticSophisticated or WikiNews
        splits: ratio of training, test and validation split, e.g. [0.7, 0.15, 0.15]
        tokenize_func: function to use for tokenization, e.g. split() or 'spacy'
        embedding_func: which embedding function to use, can be 'glove' or 'fasttext'
        batch_size: int which denotes the size of each batch
    """
    if tokenize_func == 'split':
        def tokenizer(x): return x.split()
    elif tokenize_func == 'spacy':
        tokenizer = 'spacy'
    else:
        raise ValueError(f"Error: Tokenizer function {tokenize_func} not supported")

    # text_fields will receive (batches of) Strings of text
    text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)
    # label_fields will receive the (numerical) labels of data points (e.g. 0, 1), i.e. use_vocab=False
    label_field = data.LabelField(use_vocab=False)

    if dataset_name == 'WikiSyntheticGeneral':
        dataset = WikiSyntheticGeneral(text_field, label_field)
    else:
        raise ValueError(f"Error: dataset_name {dataset_name} is not valid!")

    if splits is None:
        splits = [0.7, 0.15, 0.15]
    else:
        splits = splits
    train_data, test_data, valid_data = dataset.split(splits)
    train_size, val_size, test_size = len(train_data), len(valid_data), len(test_data)
    sizes = (train_size + val_size + test_size, train_size, val_size, test_size)

    emb_func = None
    if embedding_func == 'glove':
        emb_func = GloVe(name='6B', dim=300)
    elif embedding_func == 'fasttext':
        emb_func = FastText(language='en')

    # TODO: use pre-built vocab files instead??
    text_field.build_vocab(dataset.text, vectors=emb_func)
    del dataset

    word_embeddings = text_field.vocab.vectors

    train_iter, test_iter, valid_iter = data.Iterator.splits(
        (train_data, test_data, valid_data), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True
    )
    iters = (train_iter, valid_iter, test_iter)

    vocab_size = len(text_field.vocab)

    return vocab_size, word_embeddings, iters, sizes
