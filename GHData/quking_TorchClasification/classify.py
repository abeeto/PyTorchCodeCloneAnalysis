import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dicts = pickle.load(f, encoding='bytes')
    return dicts

data = unpickle("cifar-10-batches/data_batch_1")
print(data[b'data'].shape)