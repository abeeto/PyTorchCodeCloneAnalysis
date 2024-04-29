import gzip
import idx2numpy
import pathlib
import urllib.request

from datasets import Dataset, Mnist

HERE = pathlib.Path().cwd()
DATA = HERE.joinpath('data')

LINKS = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
]


def get_data():
    for link in LINKS:
        filename = link.split('/')[-1]
        filepath = DATA.joinpath(filename)
        urllib.request.urlretrieve(link, filename=filepath)


def load_data():
    data = {}
    for key in ['*train-images*', '*train-labels*', '*t10k-images*', '*t10k-labels*']:
        filename = [*DATA.glob(key)][0]
        with gzip.open(filename, 'rb') as file:
            data[key] = idx2numpy.convert_from_file(file)
    mnist_data = Dataset(
        data['*train-images*'], data['*train-labels*'], data['*t10k-images*'], data['*t10k-labels*']
    )
    return Mnist(mnist_data)


if __name__ == '__main__':
    get_data()
