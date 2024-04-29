import numpy as np
import cPickle
import os

CIFAR_PATH = '../cifar-100-python/'


def cifar100(seed):
    """
    Load cifar 100 data
    """
    np.random.seed(seed)

    with open(CIFAR_PATH + 'train', 'rb') as fo:
        train_data = cPickle.load(fo)

        train_x = train_data['data']
        train_y = np.array(train_data['fine_labels'])

    with open(CIFAR_PATH + 'test', 'rb') as fo:
        test_data = cPickle.load(fo)

        test_x = test_data['data']
        test_y = np.array(test_data['fine_labels'])

    # select categories
    selected_cats = np.random.choice(np.arange(0, 100), 4)
    selected_train = np.isin(train_y, selected_cats)
    train_x = train_x[selected_train]
    train_y = train_y[selected_train]
    _, train_y = np.unique(train_y, return_inverse=True)
    selected_test = np.isin(test_y, selected_cats)
    test_x = test_x[selected_test]
    test_y = test_y[selected_test]
    _, test_y = np.unique(test_y, return_inverse=True)

    train_x = scale(train_x)
    test_x = scale(test_x)

    train_x = train_x.reshape((train_x.shape[0], 3, 32, 32))
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))

    train_x = train_x.transpose(0, 2, 3, 1)
    test_x = test_x.transpose(0, 2, 3, 1)

    return (train_x, train_y), (test_x, test_y)


def scale(x):
    """
    Scale data to be between -0.5 and 0.5
    """
    x = x.astype('float32') / 255.
    x = x - 0.5
    x = x.reshape(-1, 32*32*3)
    return x


def load_data(file_name, seq_len, dictionary={}, word_i=1):
    labels = []
    sentences = []

    with open(file_name) as f:
        lines = f.readlines()

        for line in lines:
            line = line.replace('\n', '')
            split = line.split(',')
            sentence = split[0]
            label = int(split[1])

            int_sentence = []

            words = sentence.split(' ')
            for i in range(seq_len):
                if i > len(words) - 1:
                    int_sentence.append(0)
                    continue

                word = words[i]
                if word in dictionary:
                    int_sentence.append(dictionary[word])
                else:
                    dictionary[word] = word_i
                    int_sentence.append(dictionary[word])
                    word_i += 1

            if len(int_sentence) == 0:
                continue

            sentences.append(int_sentence)
            labels.append(label)

    print ("Vocabulary size: " + str(word_i))

    sentences = np.stack(sentences)
    labels = np.stack(labels)

    return sentences, labels, dictionary, word_i


def load_text_dataset(path, seq_len):
    train_x, train_y, d, word_i = load_data(
        os.path.join(path, 'train.csv'), seq_len)
    test_x, test_y, _, _ = load_data(
        os.path.join(path, 'test.csv'), seq_len, d, word_i)

    return (train_x, train_y), (test_x, test_y)
