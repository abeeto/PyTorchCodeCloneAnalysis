import numpy as np
import scipy.io as sio
import os
from random import shuffle


def load_data(data_dir, label_path):
    """
    Loads the data as two dictionaries so that it can be used by the generator class.
    :param data_dir: Directory to where the torch tensors of the data are stored
    :param label_path: Path to the .mat file of the image labels
    :return: Dictionary objects containing the train/ test partitions (the contents of which are randomly assigned
    each time this function is called. Uses an 80/20 split. Also returns a dictionary object that maps each tensor
    file to its label.
    """
    # Create a list of image IDs
    image_ids = os.listdir(data_dir)
    image_ids.sort()

    labels = sio.loadmat(label_path)
    labels = labels['labels']
    np.save('./labelsArray', labels)
    labels = np.ndarray.tolist(labels)[0]

    x_y_pairs = list(zip(image_ids, labels))
    shuffle(x_y_pairs)

    training_pairs = x_y_pairs[:int(0.8*len(x_y_pairs))]
    test_pairs = x_y_pairs[int(0.8*len(x_y_pairs)):]

    train_image_names = list(list(zip(*training_pairs))[0])
    test_image_names = list(list(zip(*test_pairs))[0])

    partition = {'train': train_image_names, 'test': test_image_names}
    labels = dict(x_y_pairs)

    training_set_size = len(train_image_names)

    print('Training set size: ' + str(len(train_image_names)))
    print('Test set size: ' + str(len(test_image_names)))

    return partition, labels, training_set_size
