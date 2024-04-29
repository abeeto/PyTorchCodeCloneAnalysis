__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from torchtext import data
from konlpy.tag import Mecab
from torchtext.data import TabularDataset, Iterator


def get_dataloader(data_path, bs, seq_len):
    mecab = Mecab()

    ID = data.Field(sequential=False,
                    use_vocab=False)

    TEXT = data.Field(sequential=True,
                      use_vocab=True,
                      tokenize=mecab.morphs,
                      lower=True,
                      batch_first=True,
                      fix_length=seq_len)

    LABEL = data.Field(sequential=False,
                       use_vocab=False,
                       is_target=True)

    train_data, test_data = TabularDataset.splits(
        path=data_path, train='ratings_train.txt', test='ratings_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)

    TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

    train_loader = Iterator(train_data, batch_size=bs)
    test_loader = Iterator(test_data, batch_size=bs)

    return train_loader, test_loader, TEXT.vocab


if __name__ == '__main__':
    train_loader, test_loader, vocab = get_dataloader('./rsc/data/', 32, 50)
    print(len(vocab.stoi))