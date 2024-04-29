import numpy as np
from sklearn.model_selection import train_test_split


class Data():
    def __init__(self):
        self.dataSize = 10000
        self.mean = [5, 7.5]
        self.std = [4, 2.5]
        self.print_stats()
        self.features = self.generate_data('features')
        self.labels = self.generate_data('labels')

    def print_stats(self):
        for i, val in enumerate(self.mean):
            print('Class {} : Mean = {} standard devidation = {}'.format(
                i, val, self.std[i]))

    def generate_data(self, type):
        if type == 'features':
            a = np.random.normal(self.mean[0], self.std[0], (5000, 512))
            b = np.random.normal(self.mean[1], self.std[1], (5000, 512))
            data = np.vstack((a, b))
            print('Shape of {} is {}'.format(type, data.shape))
            return (data.astype(np.float32))

        elif type == 'labels':
            a = np.zeros(5000)
            b = np.ones(5000)
            data = np.append(a, b)
            print('Shape of {} is {}'.format(type, data.shape))
            return (data.astype(np.float))

    def getData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                            self.labels,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            random_state=42)
        X_train = np.array([x.reshape(1, len(x)) for x in X_train])
        X_test = np.array([x.reshape(1, len(x)) for x in X_test])
        return (X_train, X_test, y_train, y_test)
